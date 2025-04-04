import os
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Flask 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 데이터 모델 (예시)
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# CSV 파일 로드
cleaned_df = pd.read_csv("cleaned_data.csv")
# 포트폴리오 최적화를 위한 주가 데이터 (각 열: 종목, 각 행: 날짜별 주가)
stock_data = pd.read_csv("stock_data.csv")

# 클러스터링에 사용할 특징 변수 (HTML 및 API 입력과 일치)
features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']

# NaN이 없는 행만 사용하여 클러스터링 수행
cleaned_df_filtered = cleaned_df[features].dropna()

# Spectral Clustering 수행
spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
labels = spectral.fit_predict(cleaned_df_filtered)

# 클러스터링 결과를 원본 데이터프레임에 할당 (인덱스 일치)
cleaned_df.loc[cleaned_df_filtered.index, 'Cluster'] = labels
cleaned_df['Cluster'] = cleaned_df['Cluster'].astype('Int64')

# 하드코딩된 클러스터 중심 (6개 변수)
centroids = pd.DataFrame({
    'PER': [28.629549, 24.969939, 22.338191, 22.468863],
    'DividendYield': [1.530, 1.805, 2.050, 1.455],
    'Beta': [1.0260, 1.0565, 1.0755, 0.9680],
    'RSI': [59.754312, 60.255378, 60.046894, 59.025935],
    'volume': [9412568, 3270264, 660531.5, 1488842],
    'Volatility': [0.014870, 0.015702, 0.016852, 0.016296]
}, index=[0, 1, 2, 3])

def map_input(cluster_feature, feature_value):
    if feature_value == 'high':
        return cluster_feature.max()
    elif feature_value == 'medium':
        return cluster_feature.mean()
    elif feature_value == 'low':
        return cluster_feature.min()
    else:
        return cluster_feature.mean()

def evaluate_cluster_fit(user_values, centroids):
    distances = []
    for idx, row in centroids.iterrows():
        distance = sum((user_values[feature] - row[feature])**2 for feature in user_values.keys())
        distances.append(distance)
    return distances.index(min(distances))

# 포트폴리오 최적화를 위한 함수들
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # 최대화를 위해 음수 반환

def get_ret_vol_sr(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    required_features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']
    if not all(feature in data for feature in required_features):
        return jsonify({"error": "Missing required features"}), 400
    for feature in required_features:
        if data[feature].lower() not in ['high', 'medium', 'low']:
            return jsonify({"error": f"Invalid value for {feature}. Must be 'high', 'medium', or 'low'."}), 400

    # 사용자 입력값 처리 및 매핑
    user_input = {feature: data[feature].lower() for feature in required_features}
    mapped_values = {feature: map_input(centroids[feature], value) for feature, value in user_input.items()}
    closest_cluster = evaluate_cluster_fit(mapped_values, centroids)
    tickers_in_cluster = cleaned_df[cleaned_df['Cluster'] == closest_cluster]['Ticker'].tolist()

    # 주가 데이터(stock_data) 처리: 결측치가 있는 열 삭제
    stock_data_clean = stock_data.dropna(axis=1)
    # 'Date' 열을 제외한 모든 열이 종목 데이터임
    ticker_columns = [col for col in stock_data_clean.columns if col != 'Date']
    # 클러스터에 해당하는 종목과 주가 데이터에 있는 종목의 교집합
    cluster_tickers = list(set(tickers_in_cluster) & set(ticker_columns))
    if not cluster_tickers:
        return jsonify({"error": "No matching tickers found in stock data for the chosen cluster."}), 400

    # 해당 종목들에 대한 주가 데이터 선택 (Date 포함)
    stock_data_cluster = stock_data_clean[['Date'] + cluster_tickers]
    # 일일 수익률 계산 (Date 열 제외)
    returns = stock_data_cluster.drop(columns=['Date']).pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # 몬테카를로 시뮬레이션을 통한 포트폴리오 최적화 (최대 샤프비율)
    num_portfolios = 10000
    all_weights = np.zeros((num_portfolios, len(mean_returns)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    for ind in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        all_weights[ind, :] = weights
        ret_vol_sr = get_ret_vol_sr(weights, mean_returns, cov_matrix)
        ret_arr[ind] = ret_vol_sr[0]
        vol_arr[ind] = ret_vol_sr[1]
        sharpe_arr[ind] = ret_vol_sr[2]
    max_sharpe_idx = sharpe_arr.argmax()
    optimal_weights = all_weights[max_sharpe_idx]
    # 포트폴리오 비중(%)로 계산: 소수점 둘째 자리까지 반올림
    optimal_portfolio = {ticker: round(weight * 100, 2) for ticker, weight in zip(cluster_tickers, optimal_weights)}

    result = {
        "closest_cluster": closest_cluster,
        "optimized_companies": tickers_in_cluster,
        "optimal_portfolio": optimal_portfolio,
        "expected_return": ret_arr[max_sharpe_idx],
        "expected_volatility": vol_arr[max_sharpe_idx]
    }
    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
