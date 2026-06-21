# Silent Trails: An Integrated OSINT & Deepfake Forensics Platform
## Final Year Project (FYP) Presentation

---

## Slide 1
**Slide Title:** Silent Trails: An Integrated OSINT & Deepfake Forensics Platform  
**Subtitle:** Profiling Human Patterns Across the Digitalverse  
**Presenter:** Faheem  

**Speaker Notes:**  
*(If the supervisor asks: "What exactly is your FYP?", this is your exact answer):*  
*"Good morning respected supervisor and panel. My final year project is called Silent Trails. It is an integrated digital forensics platform that provides two major tools: an OSINT timeline for tracking digital footprints, and a Deepfake Forensics AI for authenticating media. It is designed to tackle modern online threats for both everyday individuals and cybersecurity experts."*

---

## Slide 2
**Slide Title:** The Modern Digital Challenge

Whether you are an investigator tracking a threat or an individual protecting your privacy, you face two massive hurdles:
1.  **Data Fragmentation:** Finding isolated data points (emails, phone numbers, IPs) to check for breaches is difficult. When these scattered data points pile up, raw text-based OSINT data becomes chaotic and hard to analyze.
2.  **The Crisis of Authenticity:** With the rise of advanced facial manipulation tools, we can no longer blindly trust the visual evidence we find online. Deepfakes and face swaps have completely undermined the reliability of digital media.

**Speaker Notes:**  
*"Why did I build Silent Trails? Because both investigators and everyday users face a two-part problem. First, people need quick access to standalone data—like an individual checking if their email was breached in a hack. But when investigations scale, that data becomes chaotic. Second, when we find visual evidence online, we struggle to authenticate it. Silent Trails provides a unified toolkit to solve these exact problems."*

---

## Slide 3
**Slide Title:** The Two Pillars of Silent Trails

Silent Trails equips users with two major, complementary toolsets:
*   **🌐 Pillar 1: Digital OSINT & Intelligence Timeline** 
    *   *Direct Lookups:* Instant querying of emails (breach checks), IPs, Domains, and Phone numbers.
    *   *Optional Normalization:* Structuring raw data into an interactive, chronological timeline with risk scoring for deep-dive investigations.
*   **🔍 Pillar 2: Deepfake Forensics** 
    *   *Authentication:* A standalone multi-stream AI engine with built-in temporal video analysis to verify visual media and detect deepfakes.

*(Note: Add a screenshot of the Silent Trails main web dashboard here).*

**Speaker Notes:**  
*"Silent Trails provides users with two major pillars. The first pillar provides rapid, direct OSINT lookups for things like data breaches and IP tracking—perfect for individuals doing a personal digital assessment. If an investigator wants to go deeper, this pillar can normalize that data into a structured Intelligence Timeline. The second pillar is our Deepfake Forensics engine, serving as a dedicated authentication tool to detect facial manipulations in images or videos."*

---

## Slide 4
**Slide Title:** Pillar 1 - OSINT Gathering & Direct Queries

*   **Standalone Lookups:** Quick, actionable intelligence for everyday users and experts.
    *   *Email Breach Checks:* Verify if your credentials have been compromised across platforms.
    *   *Infrastructure Lookups:* Instant IP address and Domain intelligence.
    *   *Phone Number Tracking:* Correlate numbers to known registries.
*   **Cross-Platform Correlation:** The system acts as a wide net, capturing raw text data across the digital economy.

**Speaker Notes:**  
*"The foundation of Pillar 1 is empowering the user with immediate answers. It is not just about building massive investigator timelines; it's also a rapid lookup tool. If a regular person wants to perform a digital assessment and check if their email has been breached, or get information on an IP address, Silent Trails finds it instantly. It acts as a highly efficient search engine for digital footprints."*

---

## Slide 5
**Slide Title:** Pillar 1 - The Intelligence Timeline (Optional Deep-Dive)

*   When an investigation scales, raw data can be converted into a normalized Timeline.
*   **The Normalization Engine:** Unstructured scrape data is parsed, cleaned, and standardized.
*   **Interactive Event Timeline:** Plots text data chronologically, allowing filtering by date, location, or event type.
*   **Risk Scoring Dashboard:** Automatically assigns risk scores based on behavioral patterns and anomaly detection.
*   **Professional PDF Reporting:** Generates court-ready, one-click exportable PDF intelligence reports.

**Speaker Notes:**  
*"Now, if a cybersecurity expert *wants* to dig deeper, they can take all that raw, isolated data and feed it into our Normalization Engine. This converts chaotic data into our Intelligence Timeline. This dashboard plots text-based events chronologically and actively calculates a Risk Score based on the target's behavior. Finally, it allows the investigator to export a polished PDF report."*

---

## Slide 6
**Slide Title:** Pillar 2 - The Media Authenticity Crisis

*   While tracking behavior, we frequently uncover visual media online. How do we know this evidence hasn't been tampered with?
*   **Current deepfake detectors fail** because they overfit to specific datasets instead of learning actual forgery signatures (like face-swap blending or temporal flickering).
*   **The Goal:** Build a highly generalized, multi-domain deepfake detection model robust to real-world degradation, serving as the ultimate deepfake verification tool.

**Speaker Notes:**  
*"This brings us to the second major pillar. During an investigation, uncovering images and videos is common. But is this visual evidence manipulated? Standard deepfake detectors fail in the real world because they memorize dataset shortcuts rather than true manipulation artifacts. I needed to build a highly generalized forensic AI from scratch to detect deepfakes and face swaps reliably."*

---

## Slide 7
**Slide Title:** Deepfake Forensics Architecture

![Architecture Diagram](C:/Users/Faheem.DESKTOP-MQLKQK1/.gemini/antigravity/brain/78cd20b7-5abf-4149-813a-cb11b2c2cc10/architecture_ppt_v2_1781687256817.png)

**Speaker Notes:**  
*"This is the architecture of my Deepfake Forensics engine. We extract and align faces, apply aggressive anti-shortcut augmentations, and pass the face through THREE parallel streams: Spatial, Frequency, and Attention. These are fused using an MLP. Notice the Temporal Transformer block—our model doesn't just look at static frames; it performs temporal analysis across video sequences before giving a final Real/Fake verdict and a manipulation heatmap."*

---

## Slide 8
**Slide Title:** Deep Dive: Spatial, Frequency, Attention & Temporal

*   **🧊 Spatial Stream (DINOv2 + LoRA):** Frozen Vision Transformer with LoRA adapters. Catches texture and facial blending artifacts.
*   **📊 Frequency Stream (2D FFT + CNN):** Converts image to log-magnitude spectrum. Catches hidden mathematical fingerprints.
*   **🎯 Attention Stream (Cross-Attention):** Focuses on critical semantic regions (eyes, mouth) and produces the explainability heatmap.
*   **🎬 Temporal Transformer (Video Analysis):** Analyzes 16-frame sequences to detect inter-frame flickering, unnatural motion, and identity drift in videos.

**Speaker Notes:**  
*"Why this complex structure? Because different manipulation techniques leave different clues. Face swaps leave spatial blending artifacts. High-quality deepfakes have flawed frequency spectrums. The attention stream generates a visual heatmap of tampered areas. And crucially, our Temporal Transformer looks across 16-frame sequences in videos to catch micro-flickering and unnatural motion that a static image analysis would miss."*

---

## Slide 9
**Slide Title:** Forcing True Generalization Across Massive Datasets

*   **Extensive Training Data (Multi-Dataset Approach):**
    1.  FaceForensics++ (c23)
    2.  DFDC (Deepfake Detection Challenge)
    3.  Modern High-Quality Video Dataset (Latest generation manipulations)
*   **Anti-Shortcut Preprocessing:** Random JPEG Recompression (q60-95) + CLAHE normalization applied dynamically to prevent memorization of dataset noise.
*   **Curriculum Learning:** 
    *   Phase 1: Easy samples.
    *   Phase 2: Mixed difficulty (blur/noise).
    *   Phase 3: Chaos mode (Heavy degradation).

**Speaker Notes:**  
*"To guarantee generalization, the model couldn't be trained on just one dataset. We trained the model across three diverse datasets: FaceForensics++ c23, the massive DFDC dataset, and a modern high-quality video dataset. Combined with Curriculum Learning and aggressive anti-shortcut data degradation, this forces the network to concentrate strictly on true manipulation artifacts rather than dataset-specific lighting or compression."*

---

## Slide 10
**Slide Title:** Performance on Training Distribution

| Metric | Score | 
| :--- | :--- | 
| **Accuracy** | 94.05% |
| **Precision** | 94.30% | 
| **Recall** | 93.80% | 
| **F1-Score** | 94.05% | 

![Confusion Matrix](C:/Users/Faheem.DESKTOP-MQLKQK1/.gemini/antigravity/brain/78cd20b7-5abf-4149-813a-cb11b2c2cc10/confusion_matrix_1781685808748.png)

**Speaker Notes:**  
*"On our primary training distributions, the model achieves excellent performance with a 94% F1-score detecting manipulations like Deepfakes and Face Swaps. More importantly, the confusion matrix shows a very low false positive rate of 5.7%. In real-world forensics, falsely accusing an authentic video of being a deepfake destroys your credibility, so maintaining high precision was my top priority."*

---

## Slide 11
**Slide Title:** ROC Curves & Cross-Dataset Performance

![ROC Curve](C:/Users/Faheem.DESKTOP-MQLKQK1/.gemini/antigravity/brain/78cd20b7-5abf-4149-813a-cb11b2c2cc10/roc_curve_1781685799265.png)

*   **Training Aggregate:** AUC 0.975
*   **Cross-Evaluation Target:** **Celeb-DF (v2)**
*   **Cross-Dataset Zero-Shot:** AUC 0.938
*   **Overall Cross-Dataset Aggregate:** AUC 0.867

**Speaker Notes:**  
*"This slide proves the model works in the real world against unseen manipulation methods. After training on our massive combined dataset, we ran a strict cross-evaluation specifically against Celeb-DF v2. While we hit 0.975 AUC on our training data, the true achievement is maintaining a 0.938 AUC on Celeb-DF v2 without any fine-tuning. This proves that our multi-stream temporal approach successfully prevented the model from overfitting."*

---

## Slide 12
**Slide Title:** Proving the Multi-Stream Concept

![Ablation Study](C:/Users/Faheem.DESKTOP-MQLKQK1/.gemini/antigravity/brain/78cd20b7-5abf-4149-813a-cb11b2c2cc10/ablation_study_1781685851193.png)

**Speaker Notes:**  
*"To mathematically justify this complex architecture, I ran an ablation study. Using just the Spatial stream yields an AUC of 0.912. Adding Frequency boosts it to 0.951. Combining all streams yields our peak performance of 0.975. This proves that Spatial, Frequency, and Attention streams capture mutually exclusive, complementary forensic signals when identifying deepfakes."*

---

## Slide 13
**Slide Title:** A Comprehensive Forensic Workflow

How the Silent Trails platform empowers users:
1.  **Personal Digital Assessments:** Individuals run quick checks on their emails and digital footprint to secure their privacy.
2.  **Chronological Deep-Dive:** Cybersecurity experts normalize text data into an Intelligence Timeline to map out associated risks.
3.  **Media Verification:** Any user can upload visual media to the **Deepfake Forensics Engine**.
4.  **Authentication:** The AI scans the media's facial data and temporal consistency, flagging deepfakes with a confidence score and a visual manipulation heatmap.

**Speaker Notes:**  
*"So how do these pillars work together? Everyday individuals can use the OSINT tools for rapid personal security checks. Investigators can use the Timeline to profile advanced behaviors. And when anyone uncovers a suspicious image or video, they can run it through the Deepfake Forensics engine to verify its authenticity. Silent Trails provides a complete, trusted digital ecosystem."*

---

## Slide 14
**Slide Title:** Full-Stack & Machine Learning Technologies

*   **Frontend:** React.js, Context API, CSS Grid/Flexbox
*   **Backend (OSINT & Timeline):** Node.js, Express, MongoDB (planned)
*   **Machine Learning (Deepfake Engine):** 
    *   PyTorch, Torchvision
    *   HuggingFace Transformers (DINOv2)
    *   PEFT (LoRA)
    *   Albumentations, RetinaFace
*   **Microservice API:** FastAPI (Connects the React frontend to the PyTorch model)

**Speaker Notes:**  
*"Building this required a diverse tech stack. The user interface is built in React. The OSINT gathering and timeline normalization runs on Node.js. Because of the heavy compute requirements, the Deepfake engine runs as a completely separate microservice built in Python using PyTorch and FastAPI, which communicates seamlessly with the Node backend."*

---

## Slide 15
**Slide Title:** Current Constraints & Next Phases

*   **Broader AI-Generated Media Detection:** Currently, the model focuses strictly on deepfakes and facial manipulations (face swaps). Future updates will expand detection to fully AI-generated images (e.g., Midjourney, DALL-E) and environments.
*   **Audio Forensics:** The current model focuses on visual artifacts and does not yet detect audio-driven lip-sync manipulations (e.g., Wav2Lip).
*   **Timeline Expansion:** Integrating dark web data sources into the risk scoring algorithm.

**Speaker Notes:**  
*"While the platform is highly functional, there is room for expansion. Our deepfake model is highly specialized in detecting facial manipulations and video anomalies. However, it does not currently analyze audio tracks for lip-sync mismatches. Future work involves integrating audio forensics, expanding the model's scope to detect fully generated AI art like Midjourney, and integrating deeper dark web sources into our timeline."*

---

## Slide 16
**Slide Title:** Summary

*   **Silent Trails** successfully provides a dual-pillar solution for everyday individuals and cybersecurity experts.
*   The **OSINT and Intelligence Timeline** empowers users with both rapid personal digital assessments and structured chronological insights.
*   The **Multi-Stream Deepfake Forensics** module (trained on FF++, DFDC, etc.) achieves state-of-the-art generalization (0.867+ Cross-Dataset AUC) to ensure digital media can be trusted.
*   The platform provides a modern, unified toolkit for navigating the complexities of deepfakes and digital privacy.

**Speaker Notes:**  
*"To conclude, Silent Trails offers a complete dual-pillar solution. By combining rapid OSINT lookups for individuals, chronological timeline analysis for experts, and a highly generalized deepfake detection engine for everyone, we provide a toolkit built specifically for the challenges of modern digital privacy and verification. Thank you, I am now open to your questions."*
