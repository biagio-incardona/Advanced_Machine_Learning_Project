from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from importlib.machinery import SourceFileLoader
from sklearn.preprocessing import MinMaxScaler
foo = SourceFileLoader("myWord2Vec",
                       "C:\\Users\\biagi\\Desktop\\university\\Second Year\\First Semester\\advanced machine learning\\prog\\Advanced_Machine_Learning_Project\\ML\\myWord2Vec.py").load_module()
import myWord2Vec as w2v

class W2VGaussianNB(GaussianNB):
    def __init__(self,
                 *,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 n_proc=15,
                 epochs=100,
                 priors=None,
                 var_smoothing=1e-9):
        super().__init__(priors=priors, var_smoothing=var_smoothing)
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.n_proc=n_proc
        self.epochs=epochs
    
    def fit(self, X, y, sample_weight=None):
        self.W2V = w2v.myWord2Vec(corpus=X,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  n_proc=self.n_proc,
                                  epochs=self.epochs)
        X = self.W2V.df_w2v_vectorize(X)
        
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        #W2V = mW2V.myW2V(self.W2V_max_features, self.ngram_range)
        X = self.W2V.df_w2v_vectorize(X)
        return super().predict(X)
        
class W2VMultinomialNB(MultinomialNB):
    def __init__(self,
                 *,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 n_proc=15,
                 epochs=100,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.n_proc=n_proc
        self.epochs=epochs
        self.scaler = MinMaxScaler()
    
    def fit(self, X, y, sample_weight=None):
        self.W2V = w2v.myWord2Vec(corpus=X,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  n_proc=self.n_proc,
                                  epochs=self.epochs)
        X = self.W2V.df_w2v_vectorize(X)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        #W2V = mW2V.myW2V(self.max_features, self.ngram_range)
        X = self.W2V.df_w2v_vectorize(X)
        X = self.scaler.transform(X)
        return super().predict(X)
        
class W2VDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 *,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 n_proc=15,
                 epochs=100,
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
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.n_proc=n_proc
        self.epochs=epochs
    
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'):
        self.W2V = w2v.myWord2Vec(corpus=X,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  n_proc=self.n_proc,
                                  epochs=self.epochs)
        X = self.W2V.df_w2v_vectorize(X)
        return super().fit(X=X, y=y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=X_idx_sorted)
    
    def predict(self, X, check_input=True):
        X = self.W2V.df_w2v_vectorize(X)
        return super().predict(X=X, check_input=check_input)
    
class W2VLinearSVC(LinearSVC):
    def __init__(self,
                 *,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 n_proc=15,
                 epochs=100,
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
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.n_proc=n_proc
        self.epochs=epochs
    
    def fit(self, X, y, sample_weight=None):
        self.W2V = w2v.myWord2Vec(corpus=X,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  n_proc=self.n_proc,
                                  epochs=self.epochs)
        X = self.W2V.df_w2v_vectorize(X)
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        X = self.W2V.df_w2v_vectorize(X)
        return super().predict(X)
    
class W2VLogisticRegression(LogisticRegression):
    def __init__(self,
                 *,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 n_proc=15,
                 epochs=100,
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
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.n_proc=n_proc
        self.epochs=epochs
    
    def fit(self, X, y, sample_weight=None):
        self.W2V = w2v.myWord2Vec(corpus=X,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  n_proc=self.n_proc,
                                  epochs=self.epochs)
        X = self.W2V.df_w2v_vectorize(X)
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        X = self.W2V.df_w2v_vectorize(X)
        return super().predict(X)
          