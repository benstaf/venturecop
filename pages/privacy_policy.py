import streamlit as st

st.set_page_config(
    page_title="Terms of Service & Privacy Policy - Venture Copilot",
    page_icon="favicon.ico",
    layout="centered"
)


# Hide the sidebar and navigation completely
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
        /* Force hide sidebar on load */
        .css-1d391kg, [data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


st.title("Terms of Service & Privacy Policy")
st.markdown("*Last updated: October 2025*")

st.markdown("---")

# Terms of Service
st.header("1. Terms of Service")

st.subheader("1.1 Service Description")
st.write("""
Venture Copilot ("the Service") is an AI-powered pitch deck analysis tool operated by ReyzHub. 
The Service analyzes startup pitch decks and provides investment recommendations based on 
market, product, and founder analysis.
""")

st.subheader("1.2 Acceptance of Terms")
st.write("""
By uploading a pitch deck or using this Service, you agree to these Terms & Conditions. 
If you do not agree, please do not use the Service.
""")

st.subheader("1.3 Use of Service")
st.write("""
- The Service is provided for informational purposes only
- Analysis results do not constitute financial or investment advice
- You must have the right to upload and share any documents you submit
- You are responsible for the accuracy of information in uploaded documents
""")

st.subheader("1.4 Intellectual Property")
st.write("""
- You retain all rights to your uploaded pitch decks
- ReyzHub retains all rights to the Service, analysis framework, and generated reports
- You may not reverse engineer, copy, or redistribute the Service
""")

st.markdown("---")

# Privacy Policy
st.header("2. Privacy Policy (GDPR Compliant)")

st.subheader("2.1 Data Controller")
st.write("""
**ReyzHub**  
Server Location: Germany (European Union)  
Contact: info@reyzhub.com
""")

st.subheader("2.2 Data We Collect")
st.write("""
When you use the Service, we may collect:

- **Uploaded Documents**: PDF pitch decks you voluntarily upload
- **Extracted Text**: Text content extracted from your pitch decks for analysis
- **Analytics Data**: Anonymous usage statistics via Umami Analytics (privacy-focused, GDPR-compliant)
- **Feedback**: Voluntary feedback submitted through our forms

We do NOT collect:
- Personal identification information (unless voluntarily provided in pitch decks)
- Email addresses (unless provided via contact form)
- Payment information
- Cookies for tracking purposes
""")

st.subheader("2.3 Legal Basis for Processing (GDPR Article 6)")
st.write("""
We process your data based on:
- **Consent** (Article 6(1)(a)): By uploading documents, you consent to analysis
- **Legitimate Interest** (Article 6(1)(f)): To provide and improve our Service
""")

st.subheader("2.4 How We Use Your Data")
st.write("""
- **Analysis**: To analyze your pitch deck and generate recommendations
- **Service Improvement**: To improve our AI models and analysis framework (anonymized)
- **Communication**: To respond to your inquiries (if you contact us)

We do NOT:
- Sell your data to third parties
- Use your data for marketing without consent
- Share identifiable information with third parties (except as required by law)
""")

st.subheader("2.5 Data Storage and Retention")
st.write("""
- Uploaded pitch decks are stored on servers in Germany (EU)
- Files are timestamped and stored in the format: `filename_YYYYMMDD_HHMM.pdf`
- We retain uploaded files for operational purposes and service improvement
- You may request deletion of your data at any time (see Your Rights below)
""")

st.subheader("2.6 Data Security")
st.write("""
We implement appropriate technical and organizational measures to protect your data:
- Server hosting in EU with GDPR-compliant providers
- Secure file storage and transmission
- Limited access to uploaded documents
- Regular security assessments
""")

st.subheader("2.7 Your Rights Under GDPR")
st.write("""
You have the right to:

- **Access**: Request a copy of your data (Article 15)
- **Rectification**: Correct inaccurate data (Article 16)
- **Erasure**: Request deletion of your data ("Right to be Forgotten", Article 17)
- **Restriction**: Limit how we process your data (Article 18)
- **Portability**: Receive your data in a structured format (Article 20)
- **Object**: Object to processing based on legitimate interests (Article 21)
- **Withdraw Consent**: Withdraw consent at any time (Article 7(3))

To exercise these rights, contact us at: **info@reyzhub.com**

We will respond within 30 days as required by GDPR.
""")

st.subheader("2.8 International Data Transfers")
st.write("""
Your data is stored and processed within the European Union (Germany). 
We do not transfer data outside the EU/EEA without appropriate safeguards.
""")

st.subheader("2.9 Third-Party Services")
st.write("""
We use the following third-party services:

- **Umami Analytics** (privacy-focused, GDPR-compliant, no cookies)
- **Google Forms** (for feedback collection, subject to Google's privacy policy)

These services have their own privacy policies and are GDPR-compliant.
""")

st.subheader("2.10 Children's Privacy")
st.write("""
The Service is not intended for users under 16 years of age. 
We do not knowingly collect data from children.
""")

st.subheader("2.11 Data Breach Notification")
st.write("""
In the event of a data breach affecting your personal data, we will notify you 
and relevant supervisory authorities within 72 hours as required by GDPR Article 33.
""")

st.subheader("2.12 Supervisory Authority")
st.write("""
If you believe your data rights have been violated, you may lodge a complaint with:

**German Data Protection Authority (BfDI)**  
Graurheindorfer Str. 153  
53117 Bonn, Germany  
Website: https://www.bfdi.bund.de
""")

st.markdown("---")

# US Compliance
st.header("3. US Privacy Compliance")

st.subheader("3.1 California Consumer Privacy Act (CCPA)")
st.write("""
If you are a California resident, you have additional rights:

- Right to know what personal information is collected
- Right to delete personal information
- Right to opt-out of sale of personal information (we do not sell data)
- Right to non-discrimination for exercising privacy rights

Contact info@reyzhub.com to exercise these rights.
""")

st.markdown("---")

# Disclaimers
st.header("4. Disclaimers and Limitations")

st.subheader("4.1 No Investment Advice")
st.write("""
The Service provides analysis and recommendations based on AI models. This is NOT:
- Professional investment advice
- A guarantee of startup success or failure
- A substitute for due diligence

Always consult qualified professionals before making investment decisions.
""")

st.subheader("4.2 Accuracy of Analysis")
st.write("""
While we strive for accuracy, AI analysis may contain errors or limitations. 
Results depend on the quality and completeness of uploaded pitch decks.
""")

st.subheader("4.3 Limitation of Liability")
st.write("""
ReyzHub is not liable for:
- Investment decisions based on Service analysis
- Errors or inaccuracies in analysis results
- Technical issues or service interruptions
- Data loss (though we implement security measures)
""")

st.markdown("---")

# Changes to Terms
st.header("5. Changes to This Policy")
st.write("""
We may update this Privacy Policy and Terms of Service periodically. 
Continued use of the Service constitutes acceptance of updated terms.
""")

st.markdown("---")

# Contact
st.header("6. Contact Information")
st.write("""
For questions about this policy, privacy concerns, or to exercise your data rights:

**Email**: info@reyzhub.com  
**Website**: https://vc.reyzhub.com  
**Response Time**: Within 30 days for GDPR requests
""")

st.markdown("---")
st.info("💡 By using Venture Copilot, you acknowledge that you have read and understood this Privacy Policy and Terms of Service.")
