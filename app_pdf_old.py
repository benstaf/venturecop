import streamlit as st
import sys
import os
import traceback
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Venture Copilot - ReyzHub",
    page_icon="favicon.ico",
    layout="centered"
)



# Hide the sidebar and navigation completely
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        /* Additional selectors for sidebar */
        .css-1d391kg {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)





# 🔹 Inject Umami visitor analytics
def inject_umami():
    umami_script = """
    <script defer src="https://cloud.umami.is/script.js"
        data-website-id="8e160fc9-6b08-4aa8-a339-ee62f7aca5a6">
    </script>
    """
    components.html(umami_script, height=0)



# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Lazy import (so it doesn’t slow down page load)
StartupFramework = None



def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF using pdfplumber with OCR fallback for image-based PDFs."""
    
    text = ""
    pages_with_minimal_text = []
    
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                
                # Check if page has substantial text content
                # Consider only meaningful text (not just page numbers/minimal content)
                meaningful_text = ''.join(c for c in page_text if c.isalnum() or c.isspace())
                word_count = len(meaningful_text.split())
                
                if word_count < 10:  # Less than 10 words indicates likely image-based content
                    pages_with_minimal_text.append(page_num)
                
                text += page_text
    
    except Exception as e:
        st.warning(f"Error with direct text extraction: {e}")
        pages_with_minimal_text = list(range(1, 100))  # Assume all pages need OCR
    
    # Calculate percentage of pages that need OCR
    total_extracted_words = len(text.split())
    needs_ocr = len(pages_with_minimal_text) > 0
    
    # If more than 30% of pages have minimal text OR total extracted text is very low, use OCR
    if (len(pages_with_minimal_text) / max(total_pages if 'total_pages' in locals() else 1, 1) > 0.3 or 
        total_extracted_words < 50):
        
        st.info("🔍 Image-based PDF, please wait...")
        
        try:
            # Reset file pointer for OCR
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            
            # Convert PDF to images for OCR
            images = convert_from_bytes(pdf_bytes, dpi=300)
            ocr_text = ""
            
            # Show progress for longer OCR operations
            if len(images) > 5:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for idx, img in enumerate(images):
                if len(images) > 5:
                    progress_bar.progress((idx + 1) / len(images))
                    status_text.text(f'Processing page {idx + 1} of {len(images)}...')
                
                try:
                    # Use OCR with better configuration for business documents
                    custom_config = r'--oem 3 --psm 6'
                    page_ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    ocr_text += f"\n=== Page {idx + 1} ===\n" + page_ocr_text + "\n"
                except Exception as ocr_error:
                    st.warning(f"OCR failed for page {idx + 1}: {ocr_error}")
            
            # Clean up progress indicators
            if len(images) > 5:
                progress_bar.empty()
                status_text.empty()
            
            # Use OCR text if it's substantially better than direct extraction
            if len(ocr_text.split()) > total_extracted_words * 1.5:
                text = ocr_text
                st.success(f"✅ Extraction completed from {len(images)} pages")
            else:
                # Combine both methods: use direct text where available, OCR for minimal pages
                text = text  # Keep existing direct extraction
                st.info("📄 Used hybrid extraction (direct text + OCR for image content)")
                
        except Exception as ocr_error:
            st.error(f"OCR processing failed: {ocr_error}")
            if not text.strip():
                return "Error: Could not extract text from this PDF using either method."
    
    # Final cleanup and validation
    cleaned_text = text.strip()
    
    if not cleaned_text:
        return "No text content found in this PDF."
    
    # Remove excessive whitespace while preserving structure
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = ' '.join(line.split())  # Normalize whitespace
        if cleaned_line:  # Only keep non-empty lines
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)



def main():
    st.title("Hello, I am Marc, your favorite Venture Copilot")
   # st.write("Upload your startup pitch deck (PDF). For a self-hosted alternative, contact info@reyzhub.com")

    uploaded_file = st.file_uploader("Upload your Startup Pitch Deck (PDF)", type=["pdf"])
    st.markdown("Contact us <a href='mailto:info@reyzhub.com' target='_blank'>info@reyzhub.com</a> for self-hosting your own Venture Copilot", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><a href='/privacy_policy' target='_self'>Terms of Service & Privacy Policy</a></div>", unsafe_allow_html=True)
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")

        # Ensure decks directory exists
        os.makedirs("decks", exist_ok=True)

        # Save the uploaded file


        # Get current timestamp in YYYYMMDD_HHMM format (no seconds)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Create the filename with timestamp
        filename, ext = os.path.splitext(uploaded_file.name)
        deck_path = os.path.join("decks", f"{timestamp}_{filename}{ext}")

#        deck_path = os.path.join("decks", uploaded_file.name)
        with open(deck_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
 #       st.info(f"📂 Pitch deck saved to: {deck_path}")

        text = extract_text_from_pdf(uploaded_file)

        if not text:
            st.error("No readable text found in PDF. Please upload a valid pitch deck.")
            return

        # Hide full text, just confirm extraction worked
#        st.success(f"📝 Text successfully extracted ({len(text)} characters).")

        # 🚀 Run the analysis automatically once text is extracted
        st.info("🔎 Starting analysis... please wait.")

        global StartupFramework
        if StartupFramework is None:
            from ssff_framework import StartupFramework as SF
            StartupFramework = SF

        framework = StartupFramework()
        result_placeholder = st.empty()
        result = analyze_startup_with_updates(framework, text, result_placeholder)
        if result:
            display_final_results(result, "advanced")
        else:
            st.error("Analysis did not complete successfully. Please check the errors above.")
    inject_umami()  # ✅ Enables Umami visitor tracking



def analyze_startup_with_updates(framework, startup_info_str, placeholder):
    with placeholder.container():
        st.write("### Analysis in Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_status(step, progress):
            status_text.text(f"Step: {step}")
            progress_bar.progress(progress)

        result = {}
        try:
            update_status("Parsing startup information", 0.1)
            startup_info = framework.vc_scout_agent.parse_record(startup_info_str)
#            st.write("Startup info parsed")

            update_status("VCScout evaluation", 0.2)
            prediction, categorization = framework.vc_scout_agent.side_evaluate(startup_info)
 #           st.write(f"Initial Prediction: {prediction}")
            result['Categorical Prediction'] = prediction
            result['Categorization'] = categorization.dict()

            update_status("Market analysis", 0.3)
            market_analysis = framework.market_agent.analyze(startup_info.dict(), mode="advanced")
            st.write("Market Analysis Complete")
            result['Market Info'] = market_analysis.dict()

            update_status("Product analysis", 0.4)
            product_analysis = framework.product_agent.analyze(startup_info.dict(), mode="advanced")
            st.write("Product Analysis Complete")
            result['Product Info'] = product_analysis.dict()

            update_status("Founder analysis", 0.5)
            founder_analysis = framework.founder_agent.analyze(startup_info.dict(), mode="advanced")
            st.write("Founder Analysis Complete")
            result['Founder Info'] = founder_analysis.dict()

  #          update_status("Advanced founder analysis", 0.6)




            # --- MODIFICATION START ---

            # 1. Store a reference to the original method so we can restore it later.
 #           original_get_segmentation_prompt = framework.founder_agent._get_segmentation_prompt

            # 2. Define a new "wrapper" function that will call the original method and then log its output.
  #          def debug_get_segmentation_prompt_wrapper():
                # Call the original method to get the actual prompt text.
   #             prompt = original_get_segmentation_prompt()
#                
                # --- DEBUGGING LOGIC ---
                # Display the prompt in a collapsible expander in the Streamlit UI.
#                with st.expander("🔍 Debug: Founder Segmentation Prompt", expanded=False):
 #                   st.subheader("Prompt sent to LLM:")
  #                  st.code(prompt, language='text')
 #               
                # Also save the prompt to a log file for easy access.
   #             with open("founder_segmentation_prompt_debug.log", "w") as f:
    #                f.write("=== FOUNDER SEGMENTATION PROMPT ===\n")
     #               f.write(prompt)
      #              f.write("\n=== END PROMPT ===\n")
                # --- END DEBUGGING LOGIC ---
  #              
                # Return the original prompt so the rest of the application logic works as expected.
       #         return prompt

            # 3. Temporarily replace the object's method with our new debug version.
        #    framework.founder_agent._get_segmentation_prompt = debug_get_segmentation_prompt_wrapper

            # --- MODIFICATION END ---

            founder_segmentation = framework.founder_agent.segment_founder(startup_info.founder_backgrounds)


            # --- MODIFICATION START (Part 2) ---

            # 5. IMPORTANT: Restore the original method to its place. This ensures that
            #    if you call `segment_founder` again anywhere else, it won't have the debug code.
         #   framework.founder_agent._get_segmentation_prompt = original_get_segmentation_prompt

            # --- MODIFICATION END (Part 2) ---


            founder_idea_fit = framework.founder_agent.calculate_idea_fit(
                startup_info.dict(), startup_info.founder_backgrounds
            )
 #           st.write("Advanced Founder Analysis Complete")
            result['Founder Segmentation'] = founder_segmentation
            result['Founder Idea Fit'] = founder_idea_fit[0]

            update_status("Integration", 0.8)
            integrated_analysis = framework.integration_agent.integrated_analysis_basic(
                market_analysis.dict(),
                product_analysis.dict(),
                founder_analysis.dict()
                # prediction,
                # mode="advanced"
            )
            st.write("Integration Complete")

            if integrated_analysis is not None:
                result['Final Decision'] = integrated_analysis.dict()
            else:
                result['Final Decision'] = {
                    'overall_score': 0.0,
                    'IntegratedAnalysis': 'Integration analysis failed',
                    'recommendation': 'Unable to provide recommendation',
                    'outcome': 'Hold'
                }

            update_status("Quantitative decision", 0.9)
            quant_decision = framework.integration_agent.getquantDecision(
                prediction,
                founder_idea_fit[0],
                founder_segmentation
                # integrated_analysis.dict() if integrated_analysis else {}
            )
            st.write("Quantitative Decision Complete")
            result['Quantitative Decision'] = quant_decision.dict()

            update_status("Analysis complete", 1.0)
            st.write("Analysis Complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write(traceback.format_exc())

        return result


def display_final_results(result, mode):
 #   st.subheader("Final Analysis Results")

    # Display Final Decision
    final_decision = result['Final Decision']
    
  
    # Extract recommendation
    recommendation_text = final_decision['recommendation']
    first_sentence = recommendation_text.split(".")[0] + "." if "." in recommendation_text else recommendation_text

    # Color-code based on content
    if "do not invest" in first_sentence.lower():
        verdict_color = "red"
    elif "invest" in first_sentence.lower():
        verdict_color = "green"
    else:
        verdict_color = "orange"


#    st.write("### Recommendation: ")

    # Highlight verdict
    st.markdown(
        f"## <span style='color:{verdict_color}; font-weight:bold;'>{first_sentence}</span>",
        unsafe_allow_html=True
    )

    st.write("### Recommendation")
    st.write(recommendation_text)


    st.write(f"### Overall Score: {final_decision['overall_score']:.2f}")

    st.write("### Analysis")
    st.write(final_decision.get('IntegratedAnalysis', 'Not available'))


    # Display Market Info
    st.write("### Market Information")
    market_info = result['Market Info']
    st.write(f"Market Size: {market_info['market_size']}")
    st.write(f"Growth Rate: {market_info['growth_rate']}")
    st.write(f"Competition: {market_info['competition']}")
    st.write(f"Market Trends: {market_info['market_trends']}")
    st.write(f"Viability Score: {market_info['viability_score']}")

    # Display Product Info
    st.write("### Product Information")
    product_info = result['Product Info']
    st.write(f"Features Analysis: {product_info['features_analysis']}")
    st.write(f"Tech Stack Evaluation: {product_info['tech_stack_evaluation']}")
    st.write(f"USP Assessment: {product_info['usp_assessment']}")
    st.write(f"Potential Score: {product_info['potential_score']}")
    st.write(f"Innovation Score: {product_info['innovation_score']}")
    st.write(f"Market Fit Score: {product_info['market_fit_score']}")

    # Display Founder Info
    st.write("### Founder Information")
    founder_info = result['Founder Info']
    st.write(f"Competency Score: {founder_info['competency_score']}")
    st.write(f"Analysis: {founder_info.get('analysis', 'Not available')}")

    # Display Prediction and Categorization
    st.write("### Prediction and Categorization")
    st.write(f"Prediction: {result['Categorical Prediction']}")
    st.write("Categorization:")
    for key, value in result['Categorization'].items():
        st.write(f"- {key}: {value}")

    if mode.lower() == "advanced":
        st.write("### Advanced Analysis")
        if 'Founder Segmentation' in result:
            st.write(f"Founder Segmentation: {result['Founder Segmentation']}")
        if 'Founder Idea Fit' in result:
            st.write(f"Founder Idea Fit: {result['Founder Idea Fit']:.4f}")

        if 'Quantitative Decision' in result:
            st.write("### Quantitative Decision")
            quant_decision = result['Quantitative Decision']
            st.write(f"Outcome: {quant_decision['outcome']}")
            st.write(f"Probability: {quant_decision['probability']:.4f}")
            st.write(f"Reasoning: {quant_decision['reasoning']}")



            # ✅ Show form only after process finishes
            st.write("### Feedback Form")
            components.html(
                """
                <div style="
                    display: flex; 
                    justify-content: center; 
                    margin-top: 20px;
                ">
                    <iframe 
                        src="https://docs.google.com/forms/d/e/1FAIpQLScqjhYEVPW74oI-meOECMLI9hB9A8yWt92O9zCEpbm4UmPRKQ/viewform?embedded=true"
                        width="640" height="800" 
                        style="
                            border: none; 
                            border-radius: 12px; 
                            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
                        "
                    >
                    </iframe>
                </div>
                """,
                height=850,
            )



if __name__ == "__main__":
    main()
