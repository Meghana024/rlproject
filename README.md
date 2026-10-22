# 🏥 Hospital Resource Allocation & Admission Decision System (MDP Optimizer)

An interactive clinical decision support application powered by a **Markov Decision Process (MDP)** and solved via **Value Iteration**. The system determines mathematically optimal patient admission actions and resource allocation policies under stochastic arrival and discharge dynamics.

---

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended — Free & Easiest)
1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Hospital MDP Optimizer"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **"Deploy!"**. Your dashboard will be live on a public URL in 1-2 minutes.

---

### Option 2: Hugging Face Spaces (Free)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select **Streamlit** as the Space SDK and choose **Public** or **Private**.
3. Clone the space repo locally or upload `app.py`, `hospital_mdp.py`, `requirements.txt`, and `.streamlit/config.toml`.
4. Hugging Face will automatically build and serve the application.

---

### Option 3: Docker Deployment
Build and run the containerized application locally or on any cloud VPS (AWS, GCP, DigitalOcean):
```bash
# Build the Docker image
docker build -t hospital-mdp-app .

# Run container on port 8501
docker run -d -p 8501:8501 --name hospital_mdp hospital-mdp-app
```
Alternatively, use Docker Compose:
```bash
docker compose up -d
```
Access the application at `http://localhost:8501`.

---

### Option 4: Render / Railway / Cloud VPS
- **Render**: Connect your GitHub repository, choose **Web Service**, set the build command to `pip install -r requirements.txt`, and start command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Railway**: Connect GitHub repo; Railway automatically detects Python, installs dependencies from `requirements.txt`, and starts the service using `Procfile`.

---

## 💻 Local Quickstart

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
Or on Windows, double-click `run_app.bat`.

---

## 📊 Features
- **Real-Time Decision Support**: Real-time state query for optimal triage action recommendation.
- **Monte Carlo Simulation**: Empirical evaluation of policy dynamics across episodes.
- **Full Policy & State Explorer**: Global search and exportable CSV of all 729 state-action pairs.
- **Convergence Visualizer**: Bellman error decay plotted on log scale.
