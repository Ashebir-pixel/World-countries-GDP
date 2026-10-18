"""
Entry point for Streamlit Cloud deployment or local launch.
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("app.py", run_name="__main__")

