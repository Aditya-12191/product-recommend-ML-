from recommender import load_data, build_recommender, recommend

def main():
    df = load_data()
    cosine_sim = build_recommender(df)
    print("📦 Product Recommendation System")
    print("Type a product name (like 'Red T-Shirt') to get similar items.\n")
    while True:
        product = input("🔍 Enter product name (or 'exit'): ")
        if product.lower() == 'exit':
            break
        recommendations = recommend(product, df, cosine_sim)
        if recommendations:
            print("✅ You might also like:")
            for r in recommendations:
                print(" -", r)
        else:
            print("❌ Product not found. Try again.")

if __name__ == "__main__":
    main()
