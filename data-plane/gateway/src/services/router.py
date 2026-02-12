class WeightedRouter:
    def __init__(self):
        self.models = {}
    
    def update_weights(self, config):
        self.models = config.get("models", {})
    
    def select_model(self):
        import random
        total_weight = sum(weight for _, weight in self.models.items())
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        for model, weight in self.models.items():
            cumulative_weight += weight
            if rand < cumulative_weight:
                return model
        return None  # Fallback in case no model is selected

def weighted_random(models, weights):
    import random
    total_weight = sum(weights)
    rand = random.uniform(0, total_weight)
    cumulative_weight = 0
    for model, weight in zip(models, weights):
        cumulative_weight += weight
        if rand < cumulative_weight:
            return model
    return None  # Fallback in case no model is selected