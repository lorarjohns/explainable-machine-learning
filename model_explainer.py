from aix360.algorithms.lime import LimeTabularExplainer
from aix360.metrics import faithfulness_metric, monotonicity_metric
import numpy as np


class ModelExplainer:
    def __init__(self, train_x, test_x, model, class_names, num_features):
        self.model = model
        self.train_x = train_x
        self.test_x = test_x
        self.class_names = class_names
        self.num_features = num_features
        self.lime_explainer = LimeTabularExplainer(
                              self.train_x.values,
                              class_names=self.class_names,
                              feature_names=self.train_x.columns)

        self.instance_idx = None

    def get_exp(self, instance_idx):

        self.exp = self.lime_explainer.explain_instance(
                        self.train_x.values[instance_idx],
                        self.model.predict_proba,
                        num_features=self.num_features,
                        labels=self.class_names)
        self.instance_idx = instance_idx
        return self.exp
   
    def explain_one(self, instance_idx, print_stats=True):
        if self.instance_idx == instance_idx: 
            exp = self.exp
        else:
            exp = self.get_exp(instance_idx)

        if print_stats:
            print(f"Explanation for class {self.class_names[0]}")
            print("\n".join(map(str, exp.as_list(label=0))))
            print("\n")
            print(f"Explanation for class {self.class_names[1]}")
            print("\n".join(map(str, exp.as_list(label=1))))
        return exp

    def show_exp(self, instance_idx):
        if self.instance_idx == instance_idx:
            exp = self.exp
        else:
            exp = self.explain_one(instance_idx)
            self.exp = exp
        
        self.exp.show_in_notebook()

    def evaluate_lime(self, instance_idx):
        if self.instance_idx == instance_idx: 
            exp = self.exp
        else:
            exp = self.get_exp(instance_idx)
            self.instance_idx = instance_idx

        predicted_class = self.model.predict(
            self.test_x.values[instance_idx].reshape(1, -1))[0]

        local_explanation = exp.local_exp[predicted_class]
        x = self.test_x.values[instance_idx]

        coefs = np.zeros(x.shape[0])

        for v in local_explanation:
            coefs[v[0]] = v[1]

        base = np.zeros(x.shape[0])

        print("Faithfulness: ", faithfulness_metric(self.model, x, coefs, base))
        print("Monotonicity: ", monotonicity_metric(self.model, x, coefs, base))