"""
Streamlit UI for the Government Schemes RAG Assistant.
"""
import os

import streamlit as st
import yaml
from answer import AnswerGenerator, AnswerResult


# Page config
st.set_page_config(
    page_title="Government Schemes RAG Assistant",
    page_icon="🇮🇳",
    layout="wide",
)


@st.cache_resource
def load_config():
    """Load configuration."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_generator(provider_name: str):
    """Get cached answer generator for a provider."""
    return AnswerGenerator(provider_name=provider_name)


def format_answer_display(result: AnswerResult) -> None:
    """Display answer with citations in Streamlit."""
    # Main answer
    if result.abstained:
        st.warning(result.answer)
    else:
        st.markdown(result.answer)
    
    # Citations
    if result.citations and not result.abstained:
        with st.expander("📚 Sources", expanded=True):
            for cite in result.citations:
                st.markdown(
                    f"**[{cite['index']}]** {cite['scheme']} — *{cite['section']}*  "
                    f"([Link]({cite['url']}))  "
                    f"`Retrieved: {cite['retrieved']}`  "
                    f"`Score: {cite['score']:.3f}`"
                )
    
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Provider", result.provider)
    with col2:
        st.metric("Best Score", f"{result.best_score:.3f}")
    with col3:
        st.metric("Chunks Retrieved", result.retrieved_chunks)
    with col4:
        st.metric("Language", result.language)
    
    if result.abstained:
        st.info(f"⚠️ Low retrieval confidence (threshold: {result.best_score:.3f}). The answer may be incomplete.")


def main():
    config = load_config()
    ui_config = config.get("ui", {})
    
    # Title
    st.title(ui_config.get("title", "🇮🇳 Government Schemes RAG Assistant"))
    
    # Sidebar - Provider selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Provider toggle
        provider_options = {
            "local": "🏠 Local (Ollama) - Private, no data leaves your machine",
        }
        if os.getenv("GROQ_API_KEY"):
            provider_options["groq"] = "☁️ Cloud (Groq) - Faster, but sends your question to Groq API"
        
        provider = st.radio(
            "LLM Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=0,
        )
        
        if "groq" not in provider_options:
            st.info("Cloud mode needs a Groq API key in .env")

        # Cloud warning
        if provider == "groq":
            st.warning(
                "⚠️ **Cloud Mode Active**\n\n"
                "Your question will be sent to Groq's servers for processing. "
                "Do not use this mode for sensitive or personal information."
            )
        
        # Language selection
        languages = ui_config.get("languages", ["English", "Hindi"])
        language = st.selectbox("Language", languages, index=0)
        
        # Show config
        with st.expander("🔧 Current Config"):
            st.json({
                "provider": provider,
                "language": language,
                "top_k": config["retrieval"]["top_k"],
                "score_threshold": config["retrieval"]["score_threshold"],
                "max_context_chars": config["retrieval"]["max_context_chars"],
            })
    
    # Initialize generator
    generator = get_generator(provider)
    
    # Main chat interface
    st.markdown("Ask questions about Indian government schemes in English or Hindi.")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "What are the benefits of PM-KISAN scheme?",
            "How to apply for Ayushman Bharat?",
            "What is the eligibility for Sukanya Samriddhi Yojana?",
            "PM-KISAN योजना के लाभ क्या हैं?",
            "आयुष्मान भारत के लिए आवेदन कैसे करें?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state.question = ex
    
    # Question input
    question = st.text_input(
        "Your Question",
        value=st.session_state.get("question", ""),
        placeholder="e.g., What are the benefits of PM-KISAN scheme?",
        key="question_input",
    )
    
    # Generate button
    if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = generator.generate(question.strip(), language)
                format_answer_display(result)
            except Exception as exc:
                st.error(f"Answer generation failed ({provider}): {exc}")
    
    # Footer
    st.markdown("---")
    st.caption(
        "Data source: [myScheme.gov.in](https://www.myscheme.gov.in) | "
        "Embeddings: BAAI/bge-m3 | "
        "Local LLM: Ollama | Cloud LLM: Groq"
    )


if __name__ == "__main__":
    main()
