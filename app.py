import validators,streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from pydantic import SecretStr
import requests
from bs4 import BeautifulSoup

# streamlit app
st.set_page_config(page_title="Summerize The Content")
st.title("Summation")
st.subheader("Summerize URL")

# get the openrouter api key and url to be summerized
with st.sidebar:
    groq_api_key=st.text_input("OpenRouter API Key",value="",type="password")
    
    model_choice = st.selectbox(
        "Select Model",
        [
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
        ],
        index=0,
        help="Choose a different model if one is rate-limited"
    )

generic_url=st.text_input("URL",label_visibility="collapsed")

if st.button("Summerize the content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("please enter a valid URL. It can may be a YT or Website URL ")
    else:
        try:
            with st.spinner("Loading content..."):
                try:
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        # Try multiple language codes for YouTube transcripts
                        try:
                            loader=YoutubeLoader.from_youtube_url(
                                generic_url,
                                add_video_info=False,
                                language=['en', 'en-US', 'en-GB']
                            )
                            docs=loader.load()
                            content = "\n\n".join([doc.page_content for doc in docs])
                        except Exception as yt_error:
                            st.error(f"YouTube transcript error. Try a different video or use a web article URL instead.")
                            st.error(f"Details: {str(yt_error)}")
                            st.stop()
                    else:
                        # Use simple requests + BeautifulSoup for web pages
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        response = requests.get(generic_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text
                        content = soup.get_text()
                        
                        # Clean up whitespace
                        lines = (line.strip() for line in content.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        content = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    if not content or len(content.strip()) < 50:
                        st.error("No meaningful content could be extracted from the URL")
                        st.stop()
                    
                    # Truncate if too long (to avoid token limits)
                    if len(content) > 10000:
                        content = content[:10000] + "..."
                        
                    st.info(f"Loaded {len(content)} characters")
                except Exception as load_error:
                    st.error(f"Failed to load content: {str(load_error)}")
                    st.stop()
                
            with st.spinner("Generating summary..."):
                llm=ChatOpenAI(
                    model=model_choice,
                    api_key=SecretStr(groq_api_key),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.3
                )
                
                prompt = f"""Provide a concise summary of the following content in approximately 300 words:

{content}

Summary:"""
                
                response = llm.invoke(prompt)
                st.success(response.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
