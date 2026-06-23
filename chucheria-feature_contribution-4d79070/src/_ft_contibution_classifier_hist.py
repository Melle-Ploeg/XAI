import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import check_array

class HistGradientBoostingClassifier(HistGradientBoostingClassifier):
    def decision_path(self, X):
        """
        Return the decision path in the gradient boosting.
        Authors: Beatriz Hernandez Angel Delgado
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The non zero values are
            each node impurity. The matrix is of CSR format.
        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        X = check_array(X, order="C", accept_sparse='csr')

        if self._estimator_type == 'regressor':
            base = self._raw_predict_init(X)[0, 0]
            class_idx = 0

        if self._estimator_type == 'classifier':
            class_idx = np.argmax(self.predict_proba(X), axis=1)
            # base = self._raw_predict_init(X)[0,class_idx]
            base = self.predict_proba(X)[0,class_idx]

        residuals = []
        explanations = []
        for estimator in self.estimators_[:, class_idx].ravel():
            tree = estimator

            for i in range(tree.tree_.node_count):

                parent_idx = None

                if i in tree.tree_.children_left:
                    parent_idx = np.argwhere(tree.tree_.children_left==i)[0][0]
                    sign = "<"

                if i in tree.tree_.children_right:
                    parent_idx = np.argwhere(tree.tree_.children_right==i)[0][0]
                    sign = ">"

                feature = tree.tree_.feature[parent_idx]
                threshold = tree.tree_.threshold[parent_idx]

                if parent_idx is not None:
                    explanations.append([feature, sign, threshold])
                else:
                    explanations.append([])

            values = tree.tree_.value.reshape((1, -1))
            values = values * self.learning_rate
            decisions = tree.decision_path(X).todense()

            indicators = []
            for i in range(len(decisions)):
                values_ = values.copy()
                values_[decisions[i].squeeze() == 0] = np.nan
                values_[0, 0] = 0.
                is_leave = tree.tree_.children_left == -1
                is_value = ~np.isnan(values_)

                idx = np.argwhere(np.logical_and(is_value, is_leave))[0, 1]
                while idx != 0:
                    is_left = idx == tree.tree_.children_left
                    is_right = idx == tree.tree_.children_right
                    father_idx = np.argwhere(np.logical_or(is_left, is_right))[0][0]
                    values_[0, idx] = values_[0, idx] - values_[0, father_idx]

                    idx = father_idx
                indicators.append(values_)

            residuals.append(np.vstack(indicators))

        residuals = np.hstack(residuals)

        return base, residuals, explanations