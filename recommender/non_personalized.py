import numpy as np


class TopPopRecommender(object):

    def fit(self, URM_train):
        item_popularity = np.ediff1d(URM_all.tocsc().indptr)
        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=5):
        recommended_items = self.popular_items[0:at]
        return recommended_items

    def predict(self, user_id, at):
        return self.recommend(user_id, at)


class RandomRecommender(object):

    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.n_items, at)
        return recommended_items

    def predict(self, user_id, at):
        return self.recommend(user_id, at)
