from settings import BASE_DIR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from importlib.machinery import SourceFileLoader
foo = SourceFileLoader(
    "myTFIDF", "{BaseDir}/Advanced_Machine_Learning_Project/ML/myTFIDF.py".format(BaseDir=BASE_DIR)
).load_module()
import myTFIDF as mtfidf


class TFIDFGaussianNB(GaussianNB):
    def __init__(self, *, tfidf_max_features=100, ngram_range=(1, 2), priors=None, var_smoothing=1e-9):
        super().__init__(priors=priors, var_smoothing=var_smoothing)
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
    
    def fit(self, X, y, sample_weight=None):
        self.tfidf = mtfidf.myTFIDF(X, self.tfidf_max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X).todense()
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        #tfidf = mtfidf.myTFIDF(self.tfidf_max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X).todense()
        return super().predict(X)

    def predict_proba(self, X):
        X = self.tfidf.df_tfidf_vectorize(X).todense()
        return super().predict_proba(X)


class TFIDFMultinomialNB(MultinomialNB):
    def __init__(self, *, tfidf_max_features=100, ngram_range=(1, 2), alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
    
    def fit(self, X, y, sample_weight=None):
        self.tfidf = mtfidf.myTFIDF(X, self.tfidf_max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        #tfidf = mtfidf.myTFIDF(self.max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict(X)

    def predict_proba(self, X):
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict_proba(X)


class TFIDFDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 *,
                 tfidf_max_features=100,
                 ngram_range=(1, 2),
                 criterion='gini',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0):
        if max_depth != None and max_depth <= 0: max_depth = None
        if max_leaf_nodes != None and max_leaf_nodes <= 0: max_leaf_nodes = None
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        class_weight=class_weight,
                        ccp_alpha=ccp_alpha
                        )
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
    
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'):
        print(self.tfidf_max_features)
        self.tfidf = mtfidf.myTFIDF(X, self.tfidf_max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().fit(X=X, y=y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=X_idx_sorted)
    
    def predict(self, X, check_input=True):
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict(X=X, check_input=check_input)

    def predict_proba(self, X):
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict_proba(X)


class TFIDFLinearSVC(LinearSVC):
    def __init__(self,
                 *,
                 tfidf_max_features=100,
                 ngram_range=(1, 2),
                 penalty='l2',
                 loss='squared_hinge',
                 dual=True,
                 tol=0.0001,
                 C=1.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None, 
                 max_iter=1000):
        super().__init__(penalty=penalty,
                         loss=loss,
                         dual=dual,
                         tol=tol,
                         C=C,
                         multi_class=multi_class,
                         fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         class_weight=class_weight,
                         verbose=verbose,
                         random_state=random_state,
                         max_iter=max_iter
                         )
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
    
    def fit(self, X, y, sample_weight=None):
        self.tfidf = mtfidf.myTFIDF(X, self.tfidf_max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict(X)


class TFIDFLogisticRegression(LogisticRegression):
    def __init__(self,
                 *,
                 tfidf_max_features=100,
                 ngram_range=(1, 2),
                 penalty='l2',
                 dual=False,
                 tol=0.0001,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver='lbfgs',
                 max_iter=100,
                 multi_class='auto',
                 verbose=0,
                 warm_start=False,
                 n_jobs=None,
                 l1_ratio=None):
        super().__init__(penalty=penalty,
                         dual=dual,
                         tol=tol,
                         C=C,
                         fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         class_weight=class_weight,
                         random_state=random_state,
                         solver=solver,
                         max_iter=max_iter,
                         multi_class=multi_class,
                         verbose=verbose,
                         warm_start=warm_start,
                         n_jobs=n_jobs,
                         l1_ratio=l1_ratio)
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
    
    def fit(self, X, y, sample_weight=None):
        self.tfidf = mtfidf.myTFIDF(X, self.tfidf_max_features, self.ngram_range)
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict(X)

    def predict_proba(self, X):
        X = self.tfidf.df_tfidf_vectorize(X)
        return super().predict_proba(X)