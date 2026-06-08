!pip install pyngrok -q
!pip install streamlit
from pyngrok import ngrok
import subprocess, time

subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(2)

ngrok.set_auth_token("YOUR_NGROK_TOKEN")

proc = subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port=8501",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
])
time.sleep(5)

tunnel = ngrok.connect(8501)
print(f"\n🔗 Demo URL: {tunnel.public_url}\n")
