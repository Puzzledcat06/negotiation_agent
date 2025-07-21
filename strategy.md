INSTALL OLLAMA AND LLAMA3 LOCALLY TO RUN THIS CODE

# Buyer Agent Strategy

## Chosen Personality
The buyer agent is designed with a **"data analyst" personality**, focusing on logic, numbers, and strategic reasoning. This personality was chosen because it aligns well with real-world negotiation principles:
- **Logic-based decisions:** Avoids emotional bias and keeps conversations concise and professional.
- **Budget-conscious:** Never exceeds the allocated budget, ensuring disciplined negotiation.
- **Persuasive but calm tone:** Uses data points and value-based arguments (e.g., market price comparisons) to justify offers.

This personality creates a trustworthy buyer character that seeks optimal deals while maintaining professionalism.

---

## Negotiation Strategy
1. **Opening Offer:**
   - Starts 25–30% below the market price to anchor the negotiation.
   - Caps the initial offer at 80% of the market price, ensuring strong positioning without appearing unreasonable.

2. **Counter-Offers:**
   - Adjusts offers in small increments, typically 5–7% per round, keeping pressure on the seller to drop prices.
   - Adapts offers dynamically using the seller's previous responses.

3. **Acceptance Criteria:**
   - Accepts a deal if the seller's price drops **≥15% below the market price**, or if it's the final two rounds and the offer is close to the budget.
   - Otherwise, the agent maintains a steady negotiation stance.

4. **LLM Integration:**
   - Uses a Large Language Model to craft professional, data-focused negotiation messages (e.g., "Let's break this down" or "Let the numbers speak").

---

## Key Insights from Testing
- **Easy Market Scenario:**  
  The agent secured a deal within 8–9 rounds, often achieving **20–25% savings** due to a flexible but strategic opening bid.
  
- **Tight Budget Scenario:**  
  In challenging cases, the agent’s budget discipline prevented overpaying, even if a deal was not reached. This demonstrated the robustness of its walk-away rule.

- **Premium Product Scenario:**  
  By combining minimal incremental offers and strong data-driven messaging, the agent consistently negotiated close to its target price while maintaining a professional tone.

Overall, testing confirmed that **anchoring low, increasing gradually, and sticking to budget rules** maximized both savings and deal success rates.
