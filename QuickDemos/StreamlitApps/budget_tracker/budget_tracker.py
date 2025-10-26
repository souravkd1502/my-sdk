import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Budget Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Database setup
DB_PATH = "budget_tracker.db"


def init_db():
    """Initialize SQLite database with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Transactions table with transaction type
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                transaction_type TEXT DEFAULT 'expense',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        
        # Budgets table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL UNIQUE,
                budget_limit REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        
        # Check if transaction_type column exists, if not add it
        c.execute("PRAGMA table_info(transactions)")
        columns = [column[1] for column in c.fetchall()]
        if 'transaction_type' not in columns:
            c.execute("ALTER TABLE transactions ADD COLUMN transaction_type TEXT DEFAULT 'expense'")
        
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")
        raise


def add_transaction(date, category, amount, description, transaction_type='expense'):
    """Add a new transaction to database with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO transactions (date, category, amount, description, transaction_type)
            VALUES (?, ?, ?, ?, ?)
        """,
            (date, category, amount, description, transaction_type),
        )
        conn.commit()
        conn.close()
        return True, "Transaction added successfully"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def update_transaction(transaction_id, date, category, amount, description, transaction_type='expense'):
    """Update an existing transaction with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            UPDATE transactions 
            SET date = ?, category = ?, amount = ?, description = ?, transaction_type = ?
            WHERE id = ?
        """,
            (date, category, amount, description, transaction_type, transaction_id),
        )
        conn.commit()
        conn.close()
        return True, "Transaction updated successfully"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def delete_transaction(transaction_id):
    """Delete a transaction from database with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
        conn.commit()
        conn.close()
        return True, "Transaction deleted successfully"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def get_all_transactions():
    """Retrieve all transactions with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Error fetching transactions: {e}")
        return pd.DataFrame()


def get_category_summary(transaction_type='expense'):
    """Get spending/income by category with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT category, SUM(amount) as total
            FROM transactions
            WHERE transaction_type = ?
            GROUP BY category
            ORDER BY total DESC
        """
        df = pd.read_sql_query(query, conn, params=(transaction_type,))
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Error fetching category summary: {e}")
        return pd.DataFrame()


def set_budget(category, budget_limit):
    """Set or update budget for a category"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT OR REPLACE INTO budgets (category, budget_limit, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """,
            (category, budget_limit),
        )
        conn.commit()
        conn.close()
        return True, "Budget set successfully"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def get_budgets():
    """Get all budget limits"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM budgets", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Error fetching budgets: {e}")
        return pd.DataFrame()


def delete_budget(category):
    """Delete a budget limit"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM budgets WHERE category = ?", (category,))
        conn.commit()
        conn.close()
        return True, "Budget deleted successfully"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def get_budget_status():
    """Get current spending vs budget for each category"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT 
                b.category,
                b.budget_limit,
                COALESCE(SUM(t.amount), 0) as spent,
                b.budget_limit - COALESCE(SUM(t.amount), 0) as remaining,
                ROUND((COALESCE(SUM(t.amount), 0) / b.budget_limit) * 100, 1) as percentage
            FROM budgets b
            LEFT JOIN transactions t ON b.category = t.category 
                AND t.transaction_type = 'expense'
                AND strftime('%Y-%m', t.date) = strftime('%Y-%m', 'now')
            GROUP BY b.category, b.budget_limit
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Error fetching budget status: {e}")
        return pd.DataFrame()


def get_monthly_summary():
    """Get spending by month (expenses only)"""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT strftime('%Y-%m', date) as month, SUM(amount) as total
        FROM transactions
        WHERE transaction_type = 'expense'
        GROUP BY month
        ORDER BY month DESC
        LIMIT 12
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_weekly_summary():
    """Get spending by week (expenses only)"""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT strftime('%Y-W%W', date) as week, SUM(amount) as total
        FROM transactions
        WHERE transaction_type = 'expense'
        GROUP BY week
        ORDER BY week DESC
        LIMIT 12
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Initialize database
init_db()

# Expense Categories
EXPENSE_CATEGORIES = [
    "Groceries",
    "Eating Out",
    "Bills",
    "Rent",
    "Subscriptions",
    "Maintenance",
    "Transport",
    "Shopping",
    "Order In",
    "House Stuff",
    "Entertainment",
    "Healthcare",
    "Personal Care",
    "Miscellaneous",
]

# Income Categories
INCOME_CATEGORIES = [
    "Salary",
    "Others",
]

# All categories combined
CATEGORIES = EXPENSE_CATEGORIES

# Category icon mapping
CATEGORY_ICONS = {
    "Groceries": "🛒",
    "Eating Out": "🍽️",
    "Bills": "💡",
    "Rent": "🏠",
    "Subscriptions": "📱",
    "Maintenance": "🔧",
    "Transport": "🚗",
    "Shopping": "🛍️",
    "Order In": "🍕",
    "House Stuff": "🏡",
    "Entertainment": "🎬",
    "Healthcare": "⚕️",
    "Personal Care": "💆",
    "Miscellaneous": "📌",
}

# Category colors for visual distinction
CATEGORY_COLORS = {
    "Groceries": "#2ecc71",
    "Eating Out": "#e74c3c",
    "Bills": "#f39c12",
    "Rent": "#9b59b6",
    "Subscriptions": "#3498db",
    "Maintenance": "#e67e22",
    "Transport": "#1abc9c",
    "Shopping": "#e91e63",
    "Order In": "#ff5722",
    "House Stuff": "#795548",
    "Entertainment": "#673ab7",
    "Healthcare": "#009688",
    "Personal Care": "#ff9800",
    "Miscellaneous": "#607d8b",
}

# Custom CSS with enhanced styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: white;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    
    .transaction-card {
        background: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .transaction-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateX(5px);
    }
    
    .category-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .amount-positive {
        color: #2ecc71;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    .amount-negative {
        color: #e74c3c;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8em;
        font-weight: 700;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    .custom-progress {
        height: 25px;
        border-radius: 12px;
        background: #e0e0e0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: 700;
        color: #667eea;
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #555;
        font-size: 0.95em;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    .category-icon {
        font-size: 1.5em;
        margin-right: 8px;
    }
    
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .empty-state-icon {
        font-size: 4em;
        margin-bottom: 20px;
    }
    
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        font-weight: 500;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.title("💰 Budget Tracker & Manager")
st.markdown("---")

# Sidebar - Add Transaction
with st.sidebar:
    st.header("➕ Add Transaction")
    
    # Transaction type selection outside form for dynamic category update
    transaction_type = st.radio("Type", ["Expense", "Income"], horizontal=True, key="trans_type")

    with st.form("add_transaction_form"):
        transaction_date = st.date_input("Date", datetime.now())
        
        # Dynamic category selection based on type
        if transaction_type == "Expense":
            category = st.selectbox("Category", EXPENSE_CATEGORIES)
        else:
            category = st.selectbox("Category", INCOME_CATEGORIES)
        
        amount = st.number_input("Amount (₹)", min_value=0.0, step=0.01, format="%.2f")
        description = st.text_input("Description")

        submitted = st.form_submit_button("Add Transaction", use_container_width=True)

        if submitted:
            if amount > 0:
                success, message = add_transaction(
                    transaction_date.strftime("%Y-%m-%d"), 
                    category, 
                    amount, 
                    description,
                    transaction_type.lower()
                )
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.error("Please enter a valid amount")

    st.markdown("---")

    # Budget Management
    st.header("💰 Set Budgets")
    with st.form("budget_form"):
        budget_category = st.selectbox("Category", EXPENSE_CATEGORIES, key="budget_cat")
        budget_limit = st.number_input("Monthly Budget (₹)", min_value=0.0, step=100.0, format="%.2f")
        
        col1, col2 = st.columns(2)
        with col1:
            set_budget_btn = st.form_submit_button("Set Budget", use_container_width=True)
        with col2:
            del_budget_btn = st.form_submit_button("Delete", use_container_width=True)
        
        if set_budget_btn and budget_limit > 0:
            success, message = set_budget(budget_category, budget_limit)
            if success:
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")
        
        if del_budget_btn:
            success, message = delete_budget(budget_category)
            if success:
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")

    st.markdown("---")

    # Export data
    st.header("📥 Export Data")
    df_all = get_all_transactions()
    if not df_all.empty:
        csv = df_all.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"budget_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No data to export yet")

# Main content
df_transactions = get_all_transactions()

if df_transactions.empty:
    st.markdown(
        """
        <div class='empty-state'>
            <div class='empty-state-icon'>💰📊</div>
            <h2 style='color: #667eea;'>Welcome to Your Budget Tracker!</h2>
            <p style='font-size: 1.1em; color: #555;'>Start tracking your expenses by adding your first transaction.</p>
            <p style='color: #888;'>👈 Use the sidebar to add a new transaction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Summary metrics with income tracking
    col1, col2, col3, col4 = st.columns(4)

    # Calculate totals by transaction type
    df_expenses = df_transactions[df_transactions.get('transaction_type', 'expense') == 'expense']
    df_income = df_transactions[df_transactions.get('transaction_type', 'expense') == 'income']
    
    total_expenses = df_expenses["amount"].sum() if not df_expenses.empty else 0
    total_income = df_income["amount"].sum() if not df_income.empty else 0
    net_savings = total_income - total_expenses

    with col1:
        st.metric("Total Expenses", f"₹{total_expenses:,.2f}", delta=None)

    with col2:
        st.metric("Total Income", f"₹{total_income:,.2f}", delta=None)

    with col3:
        delta_color = "normal" if net_savings >= 0 else "inverse"
        st.metric("Net Savings", f"₹{net_savings:,.2f}", 
                 delta=f"{'Surplus' if net_savings >= 0 else 'Deficit'}")

    with col4:
        top_expense_cat = (
            get_category_summary('expense').iloc[0]["category"]
            if not get_category_summary('expense').empty
            else "N/A"
        )
        st.metric("Top Expense", top_expense_cat)

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "📅 Weekly", "📆 Monthly", "📋 Transactions"]
    )

    with tab1:
        # Budget Status Section
        df_budget_status = get_budget_status()
        if not df_budget_status.empty:
            st.subheader("📊 Budget Status (This Month)")
            
            for idx, row in df_budget_status.iterrows():
                category_color = CATEGORY_COLORS.get(row['category'], "#667eea")
                
                # Determine status color
                if row['percentage'] >= 90:
                    status_color = "#e74c3c"  # Red
                    status_text = "⚠️ Over Budget"
                elif row['percentage'] >= 70:
                    status_color = "#f39c12"  # Yellow
                    status_text = "⚡ Approaching Limit"
                else:
                    status_color = "#2ecc71"  # Green
                    status_text = "✓ On Track"
                
                st.markdown(
                    f"""
                    <div style='background: white; padding: 15px; border-radius: 10px; margin: 10px 0; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid {category_color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                            <div>
                                <span style='font-size: 1.2em; font-weight: 600; color: #2c3e50;'>{row['category']}</span>
                                <span style='margin-left: 10px; color: {status_color}; font-weight: 600;'>{status_text}</span>
                            </div>
                            <div style='text-align: right;'>
                                <div style='font-size: 0.9em; color: #7f8c8d;'>₹{row['spent']:,.2f} / ₹{row['budget_limit']:,.2f}</div>
                                <div style='font-weight: 700; color: {status_color};'>{row['percentage']}%</div>
                            </div>
                        </div>
                        <div style='background: #e0e0e0; height: 10px; border-radius: 5px; overflow: hidden;'>
                            <div style='background: {status_color}; width: {min(row["percentage"], 100)}%; height: 100%;'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            st.markdown("---")
        
        st.subheader("Category Breakdown")

        df_category = get_category_summary()

        if not df_category.empty:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Pie chart with custom colors
                colors = [CATEGORY_COLORS.get(cat, "#667eea") for cat in df_category["category"]]
                fig_pie = px.pie(
                    df_category,
                    values="total",
                    names="category",
                    title="Spending by Category",
                    hole=0.4,
                    color_discrete_sequence=colors,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    textfont_size=12,
                    marker=dict(line=dict(color='white', width=2))
                )
                fig_pie.update_layout(
                    font=dict(family="Inter, sans-serif", size=14),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Category table with icons
                st.markdown("### Category Totals")
                for idx, row in df_category.iterrows():
                    percentage = (row["total"] / total_expenses) * 100 if total_expenses > 0 else 0
                    category_icon = CATEGORY_ICONS.get(row['category'], "📌")
                    category_color = CATEGORY_COLORS.get(row['category'], "#667eea")
                    
                    st.markdown(
                        f"""
                        <div style='margin: 15px 0;'>
                            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                                <span style='font-size: 1.5em;'>{category_icon}</span>
                                <span style='font-weight: 600; color: #2c3e50;'>{row['category']}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(percentage / 100)
                    st.markdown(
                        f"<div style='color: {category_color}; font-weight: 600;'>₹{row['total']:,.2f} ({percentage:.1f}%)</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

        # Bar chart with custom colors
        st.markdown("---")
        if not df_category.empty:
            colors_bar = [CATEGORY_COLORS.get(cat, "#667eea") for cat in df_category["category"]]
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=df_category["category"],
                    y=df_category["total"],
                    marker_color=colors_bar,
                    text=df_category["total"].apply(lambda x: f"₹{x:,.0f}"),
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'
                )
            ])
            fig_bar.update_layout(
                title="Spending by Category (Bar Chart)",
                xaxis_title="Category",
                yaxis_title="Amount (₹)",
                font=dict(family="Inter, sans-serif", size=14),
                showlegend=False,
                hovermode='x',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            fig_bar.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Weekly Spending Trends")

        df_weekly = get_weekly_summary()

        if not df_weekly.empty:
            # Line chart for weekly trends with gradient
            fig_weekly = go.Figure()
            fig_weekly.add_trace(
                go.Scatter(
                    x=df_weekly["week"],
                    y=df_weekly["total"],
                    mode="lines+markers",
                    name="Weekly Spending",
                    line=dict(color="#667eea", width=4),
                    marker=dict(size=10, color="#764ba2", line=dict(color='white', width=2)),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    hovertemplate='<b>Week: %{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'
                )
            )
            fig_weekly.update_layout(
                title="Weekly Spending Trend",
                xaxis_title="Week",
                yaxis_title="Amount (₹)",
                hovermode="x unified",
                font=dict(family="Inter, sans-serif", size=14),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_weekly, use_container_width=True)

            # Weekly summary table
            st.markdown("### Weekly Breakdown")
            df_weekly_display = df_weekly.copy()
            df_weekly_display["total"] = df_weekly_display["total"].apply(
                lambda x: f"₹{x:,.2f}"
            )
            st.dataframe(df_weekly_display, use_container_width=True, hide_index=True)
        else:
            st.info("No weekly data available yet")

    with tab3:
        st.subheader("Monthly Spending Trends")

        df_monthly = get_monthly_summary()

        if not df_monthly.empty:
            # Bar chart for monthly trends with gradient
            fig_monthly = go.Figure(data=[
                go.Bar(
                    x=df_monthly["month"],
                    y=df_monthly["total"],
                    marker=dict(
                        color=df_monthly["total"],
                        colorscale=[[0, '#667eea'], [1, '#764ba2']],
                        line=dict(color='white', width=1)
                    ),
                    text=df_monthly["total"].apply(lambda x: f"₹{x:,.0f}"),
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'
                )
            ])
            fig_monthly.update_layout(
                title="Monthly Spending Trend",
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                font=dict(family="Inter, sans-serif", size=14),
                showlegend=False,
                hovermode='x',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Monthly summary table
            st.markdown("### Monthly Breakdown")
            df_monthly_display = df_monthly.copy()
            df_monthly_display["total"] = df_monthly_display["total"].apply(
                lambda x: f"₹{x:,.2f}"
            )
            st.dataframe(df_monthly_display, use_container_width=True, hide_index=True)
        else:
            st.info("No monthly data available yet")

    with tab4:
        st.subheader("All Transactions")

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_category = st.multiselect(
                "Filter by Category", CATEGORIES, default=None
            )

        with col2:
            date_from = st.date_input("From Date", datetime.now() - timedelta(days=30))

        with col3:
            date_to = st.date_input("To Date", datetime.now())

        # Apply filters
        df_filtered = df_transactions.copy()

        if filter_category:
            df_filtered = df_filtered[df_filtered["category"].isin(filter_category)]

        df_filtered["date"] = pd.to_datetime(df_filtered["date"])
        df_filtered = df_filtered[
            (df_filtered["date"] >= pd.to_datetime(date_from))
            & (df_filtered["date"] <= pd.to_datetime(date_to))
        ]

        # Display transactions with delete option
        st.markdown(f"**Showing {len(df_filtered)} transactions**")
        st.markdown("")

        for idx, row in df_filtered.iterrows():
            category_color = CATEGORY_COLORS.get(row["category"], "#667eea")
            category_icon = CATEGORY_ICONS.get(row["category"], "📌")
            
            # Create a styled transaction card
            st.markdown(
                f"""
                <div style='
                    background: white;
                    padding: 15px 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-left: 5px solid {category_color};
                    transition: all 0.3s ease;
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='display: flex; align-items: center; gap: 15px;'>
                            <span style='font-size: 2em;'>{category_icon}</span>
                            <div>
                                <div style='font-weight: 600; color: #2c3e50; font-size: 1.1em;'>{row['category']}</div>
                                <div style='color: #7f8c8d; font-size: 0.9em;'>{row['date']}</div>
                            </div>
                        </div>
                        <div style='text-align: right;'>
                            <div style='font-weight: 700; color: {category_color}; font-size: 1.3em;'>₹{row['amount']:,.2f}</div>
                            <div style='color: #95a5a6; font-size: 0.9em; font-style: italic;'>{row['description'] if pd.notna(row['description']) else 'No description'}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Edit and Delete buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with st.expander("✏️ Edit", expanded=False):
                    with st.form(key=f"edit_form_{row['id']}"):
                        edit_date = st.date_input("Date", value=pd.to_datetime(row['date']).date(), key=f"edit_date_{row['id']}")
                        
                        # Get current transaction type
                        current_type = row.get('transaction_type', 'expense')
                        edit_type = st.radio("Type", ["Expense", "Income"], 
                                           index=0 if current_type == 'expense' else 1,
                                           key=f"edit_type_{row['id']}", horizontal=True)
                        
                        # Dynamic category based on type
                        if edit_type == "Expense":
                            edit_category = st.selectbox("Category", EXPENSE_CATEGORIES, 
                                                        index=EXPENSE_CATEGORIES.index(row['category']) if row['category'] in EXPENSE_CATEGORIES else 0,
                                                        key=f"edit_cat_{row['id']}")
                        else:
                            edit_category = st.selectbox("Category", INCOME_CATEGORIES,
                                                        index=INCOME_CATEGORIES.index(row['category']) if row['category'] in INCOME_CATEGORIES else 0,
                                                        key=f"edit_cat_{row['id']}")
                        
                        edit_amount = st.number_input("Amount (₹)", value=float(row['amount']), 
                                                     min_value=0.0, step=0.01, format="%.2f",
                                                     key=f"edit_amt_{row['id']}")
                        edit_desc = st.text_input("Description", value=row['description'] if pd.notna(row['description']) else "",
                                                 key=f"edit_desc_{row['id']}")
                        
                        if st.form_submit_button("💾 Save Changes", use_container_width=True):
                            if edit_amount > 0:
                                success, message = update_transaction(
                                    row['id'],
                                    edit_date.strftime("%Y-%m-%d"),
                                    edit_category,
                                    edit_amount,
                                    edit_desc,
                                    edit_type.lower()
                                )
                                if success:
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                st.error("Amount must be greater than 0")
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", key=f"del_{row['id']}", use_container_width=True):
                    success, message = delete_transaction(row["id"])
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Budget Tracker - Manage your expenses efficiently 💰</p>
    </div>
""",
    unsafe_allow_html=True,
)
