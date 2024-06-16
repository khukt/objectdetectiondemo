import streamlit as st

def main():
    st.title("Teachable Machine with Streamlit")

    st.markdown("""
        <iframe src="static/index.html" width="100%" height="700px" frameborder="0"></iframe>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
