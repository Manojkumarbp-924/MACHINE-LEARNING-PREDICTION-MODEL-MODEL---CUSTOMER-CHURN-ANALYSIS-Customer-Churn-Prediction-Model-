import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time
from typing import List, Dict
import re

# Page configuration
st.set_page_config(page_title="AI News Agent", page_icon="📰", layout="wide")

# Initialize session state
if 'news_data' not in st.session_state:
    st.session_state.news_data = []
if 'ollama_available' not in st.session_state:
    st.session_state.ollama_available = False

class NewsAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama2"):
        self.ollama_url = ollama_url
        self.model = model
        self.news_sources = {
            "BBC": "https://www.bbc.com/news",
            "Reuters": "https://www.reuters.com",
            "The Guardian": "https://www.theguardian.com/international"
        }
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def scrape_news(self, url: str, source_name: str) -> List[Dict]:
        """Scrape news articles from a given URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            
            # Generic scraping - looks for common article patterns
            headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], limit=20)
            
            for headline in headlines:
                text = headline.get_text().strip()
                if len(text) > 20 and len(text) < 200:  # Filter out too short/long texts
                    link = headline.find_parent('a')
                    url_link = link['href'] if link and link.get('href') else ""
                    
                    # Make relative URLs absolute
                    if url_link and not url_link.startswith('http'):
                        from urllib.parse import urljoin
                        url_link = urljoin(url, url_link)
                    
                    articles.append({
                        'title': text,
                        'url': url_link,
                        'source': source_name,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return articles[:15]  # Return top 15 to filter down to 10
        except Exception as e:
            st.error(f"Error scraping {source_name}: {str(e)}")
            return []
    
    def summarize_with_ollama(self, articles: List[Dict]) -> str:
        """Use Ollama to summarize news articles"""
        try:
            # Prepare articles text
            articles_text = "\n\n".join([
                f"{i+1}. {article['title']} (Source: {article['source']})"
                for i, article in enumerate(articles[:10])
            ])
            
            prompt = f"""You are a news summarization assistant. Below are today's top news headlines from various sources. 
            
Please provide a concise summary of the top 10 most important news items, organizing them by relevance and importance. For each news item, provide:
- A brief summary (2-3 sentences)
- Why it's significant

Headlines:
{articles_text}

Provide a well-structured summary:"""
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Unable to generate summary (Status: {response.status_code})"
                
        except Exception as e:
            return f"Error summarizing with Ollama: {str(e)}"
    
    def send_email(self, recipient: str, subject: str, body: str, 
                   sender_email: str, sender_password: str, smtp_server: str = "smtp.gmail.com", 
                   smtp_port: int = 587) -> bool:
        """Send email with news summary"""
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            st.error(f"Error sending email: {str(e)}")
            return False

# UI Layout
st.title("📰 AI News Summarization Agent")
st.markdown("*Powered by Ollama & Streamlit*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama settings
    st.subheader("Ollama Settings")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
    ollama_model = st.text_input("Model Name", "llama2")
    
    # Check Ollama connection
    if st.button("Check Ollama Connection"):
        agent = NewsAgent(ollama_url, ollama_model)
        if agent.check_ollama_connection():
            st.success("✅ Ollama is running!")
            st.session_state.ollama_available = True
        else:
            st.error("❌ Cannot connect to Ollama. Make sure it's running.")
            st.session_state.ollama_available = False
    
    st.divider()
    
    # Email settings
    st.subheader("Email Settings")
    sender_email = st.text_input("Sender Email")
    sender_password = st.text_input("Email Password/App Password", type="password")
    recipient_email = st.text_input("Recipient Email")
    
    st.info("📌 For Gmail, use an App Password: [Generate here](https://myaccount.google.com/apppasswords)")
    
    st.divider()
    
    # Scheduling
    st.subheader("Schedule Settings")
    schedule_time = st.time_input("Daily Summary Time", value=None)
    
    if st.button("💾 Save Configuration"):
        st.success("Configuration saved!")

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Fetch News", "📧 Generate & Send", "ℹ️ Info"])

with tab1:
    st.header("Fetch Latest News")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Click the button below to scrape news from multiple sources:")
    
    with col2:
        if st.button("🔄 Fetch News", type="primary"):
            agent = NewsAgent(ollama_url, ollama_model)
            
            with st.spinner("Fetching news from sources..."):
                all_articles = []
                progress_bar = st.progress(0)
                
                sources = agent.news_sources
                for idx, (source_name, url) in enumerate(sources.items()):
                    st.write(f"Scraping {source_name}...")
                    articles = agent.scrape_news(url, source_name)
                    all_articles.extend(articles)
                    progress_bar.progress((idx + 1) / len(sources))
                
                # Remove duplicates based on title similarity
                unique_articles = []
                seen_titles = set()
                for article in all_articles:
                    title_lower = article['title'].lower()
                    if title_lower not in seen_titles:
                        unique_articles.append(article)
                        seen_titles.add(title_lower)
                
                st.session_state.news_data = unique_articles[:10]
                st.success(f"✅ Fetched {len(st.session_state.news_data)} unique articles!")
    
    # Display fetched news
    if st.session_state.news_data:
        st.subheader(f"Top {len(st.session_state.news_data)} News Articles")
        
        for idx, article in enumerate(st.session_state.news_data):
            with st.expander(f"{idx+1}. {article['title']}"):
                st.write(f"**Source:** {article['source']}")
                if article['url']:
                    st.write(f"**URL:** {article['url']}")
                st.write(f"**Fetched:** {article['timestamp']}")

with tab2:
    st.header("Generate Summary & Send Email")
    
    if not st.session_state.news_data:
        st.warning("⚠️ No news data available. Please fetch news first from the 'Fetch News' tab.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Generate AI Summary", type="primary"):
                if not st.session_state.ollama_available:
                    st.error("❌ Ollama connection not verified. Please check connection in sidebar.")
                else:
                    agent = NewsAgent(ollama_url, ollama_model)
                    
                    with st.spinner("Generating AI summary... This may take a minute."):
                        summary = agent.summarize_with_ollama(st.session_state.news_data)
                        st.session_state.summary = summary
                        st.success("✅ Summary generated!")
        
        with col2:
            if st.button("📧 Send Email"):
                if not all([sender_email, sender_password, recipient_email]):
                    st.error("❌ Please configure email settings in the sidebar.")
                elif 'summary' not in st.session_state:
                    st.error("❌ Please generate a summary first.")
                else:
                    agent = NewsAgent(ollama_url, ollama_model)
                    
                    subject = f"Daily News Summary - {datetime.now().strftime('%B %d, %Y')}"
                    
                    with st.spinner("Sending email..."):
                        success = agent.send_email(
                            recipient_email,
                            subject,
                            st.session_state.summary,
                            sender_email,
                            sender_password
                        )
                        
                        if success:
                            st.success(f"✅ Email sent to {recipient_email}!")
                        else:
                            st.error("❌ Failed to send email. Check your credentials.")
        
        # Display summary
        if 'summary' in st.session_state:
            st.divider()
            st.subheader("📝 Generated Summary")
            st.markdown(st.session_state.summary)

with tab3:
    st.header("ℹ️ Setup Instructions")
    
    st.markdown("""
    ### Prerequisites
    
    1. **Install Ollama**
       ```bash
       # Visit https://ollama.ai to download and install
       # After installation, pull a model:
       ollama pull llama2
       ```
    
    2. **Install Required Python Packages**
       ```bash
       pip install streamlit requests beautifulsoup4 schedule
       ```
    
    ### How to Use
    
    1. **Configure Ollama**: Enter your Ollama URL and model name in the sidebar
    2. **Check Connection**: Verify Ollama is running
    3. **Configure Email**: Set up your email credentials (use App Password for Gmail)
    4. **Fetch News**: Go to the "Fetch News" tab and click "Fetch News"
    5. **Generate Summary**: In the "Generate & Send" tab, click "Generate AI Summary"
    6. **Send Email**: Click "Send Email" to receive the summary
    
    ### Gmail App Password Setup
    
    1. Go to Google Account settings
    2. Enable 2-Factor Authentication
    3. Generate an App Password at https://myaccount.google.com/apppasswords
    4. Use this App Password instead of your regular password
    
    ### Running the Application
    
    ```bash
    streamlit run app.py
    ```
    
    ### For Daily Automation
    
    To run this daily automatically, you can:
    - Use a task scheduler (Windows Task Scheduler, cron on Linux)
    - Deploy on a cloud service with scheduled jobs
    - Use a separate Python script with the schedule library
    """)

# Footer
st.divider()
st.markdown("*Made with ❤️ using Streamlit and Ollama*")