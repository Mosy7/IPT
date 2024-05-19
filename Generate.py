# Function to get top N recommendations for a given user
def get_top_n_recommendations(user_id, n=10):
    user_ratings = trainset.ur[trainset.to_inner_uid(user_id)]
    items = [item_id for (item_id, _) in user_ratings]
    recommendations = []

    for item_id in trainset.all_items():
        if item_id not in items:
            est_rating = algo.predict(user_id, trainset.to_raw_iid(item_id)).est
            recommendations.append((item_id, est_rating))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# Example usage for user with ID 1
user_id = 1
top_n_recommendations = get_top_n_recommendations(user_id, n=10)
print(f"Top 10 recommendations for user {user_id}: {top_n_recommendations}")