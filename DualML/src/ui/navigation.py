# src/ui/navigation.py
import streamlit as st

def render_sidebar_nav(current_label: str):
    """Render the consistent Dashboard-like sidebar navigation on any page.
    current_label: the label of the current page to mark active.
    """
    # Hide Streamlit default multipage sidebar nav
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Dashboard")

    nav_items = [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🤖 Model Training",
        "📈 Predictions",
        "⚙️ Admin Panel",
    ]

    # Preserve state across pages
    if "nav" not in st.session_state:
        st.session_state.nav = current_label if current_label in nav_items else nav_items[0]

    # Render buttons and navigate on click
    for item in nav_items:
        is_active = (current_label == item)
        if st.sidebar.button(
            item,
            key=f"global_nav_{hash(item)}",
            type=("primary" if is_active else "secondary"),
            use_container_width=True,
        ):
            st.session_state.nav = item
            # Route to target page
            target = {
                "🏠 Dashboard": "app.py",
                "📊 Data Analysis": "pages/1_📊_Data_Analysis.py",
                "🤖 Model Training": "pages/2_🤖_Model_Training.py",
                "📈 Predictions": "pages/3_📈_Predictions.py",
                "⚙️ Admin Panel": "pages/4_⚙️_Admin_Panel.py",
            }[item]
            st.switch_page(target)
