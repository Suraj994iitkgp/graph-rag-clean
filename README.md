# Graph-RAG Based Chatbot for Car Brake System Analysis

We have developed a **Graph-RAG based chatbot** that helps answer questions related to **understanding the input-output relationships in a car’s brake system**. This chatbot is essential for modeling causal relationships between the driver, sensors, and environmental factors affecting braking performance and system states, useful for detection, diagnostics, and recommendation purposes in repair manuals.

## Inputs

- **Driver and User Inputs:** Brake pedal force/position, brake switch signals, and driver behavior.
- **Sensor Inputs:** Wheel speed sensors, brake fluid pressure sensors, pedal travel sensors, ABS wheel speed sensors, yaw rate sensors, and vehicle speed sensors.
- **Environmental Inputs:** Road conditions (wet, icy, dry), slope/gradient, temperature, and vehicle load.
- **System Status Signals:** Electronic control unit (ECU) states, ABS/EBD/ESC activation flags, brake lining wear sensors.

## Outputs

- **Actuation Outputs:** Hydraulic pressure to brake calipers or wheel cylinders, electronic brake force modulation commands for ABS/EBD/ESC, brake light activation.
- **System Feedback or Alerts:** Diagnostic trouble codes, warning lights on the dashboard, system error notifications.

## Technologies Used

- **Vector Database:** Pinecone for managing vector data.
- **Generator Model:** Flan-T5 (small/large) via Hugging Face transformers.
- **Retriever:** Graph RAG (Retrieval-Augmented Generation).
- **Hosting:** Simple HTML-based frontend for hosting the chatbot.

## Screenshot

![Chatbot Screenshot](images/screenshot.png)

---

Feel free to explore the project and reach out for any questions!
