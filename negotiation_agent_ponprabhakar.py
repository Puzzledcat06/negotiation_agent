import json
import random
import time
import argparse
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# ===============================
# CONFIGURATION
# ===============================
USE_LLM = True
LLM_MODEL_NAME = "llama3"  # Or "llama3:8b-q4_K_M"
LLM_SYSTEM_PROMPT = (
    "You are a calm, data-driven buyer. "
    "You analyze offers logically and speak in a concise, professional tone. "
    "Use phrases like 'Let's break this down' or 'Let the numbers speak'. "
    "Avoid emotional language, and never exceed your budget."
)
LLM_URL = "http://localhost:11434/v1"
LLM_API_KEY = "ollama"

# ===============================
# LLM CALL & PARSING HELPERS
# ===============================
def call_llm(prompt: str, model: str = LLM_MODEL_NAME) -> str:
    import openai
    openai.api_base = LLM_URL
    openai.api_key = LLM_API_KEY
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.45
    )
    return response['choices'][0]['message']['content'].strip()

def extract_offer_from_json(llm_reply: str, budget: int, fallback: int) -> Tuple[int, str]:
    match = re.search(r'\{.*?\}', llm_reply, re.DOTALL)
    try:
        parsed = json.loads(match.group() if match else llm_reply)
        offer = int(parsed.get("offer", fallback))
        offer = min(offer, budget)
        msg = parsed.get("message", "Here's my revised offer.")
        return offer, msg
    except Exception:
        match = re.search(r"\b\d{5,7}\b", llm_reply.replace(",", ""))
        offer = int(match.group()) if match else fallback
        offer = min(offer, budget)
        return offer, llm_reply if llm_reply else f"Fallback offer: ₹{offer}"

def clamp_offer(offer, context, round_num):
    market = context.product.base_market_price
    if round_num == 1:
        max_offer = int(market * random.uniform(0.75, 0.82))  # slightly randomized first cap
    elif round_num <= 4:
        max_offer = int(market * random.uniform(0.85, 0.93))
    else:
        max_offer = context.your_budget
    return min(offer, max_offer)

# ===============================
# DATA STRUCTURES
# ===============================
@dataclass
class Product:
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    product: Product
    your_budget: int
    current_round: int
    seller_offers: List[int]
    your_offers: List[int]
    messages: List[Dict[str, str]]

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ===============================
# BASE AGENT
# ===============================
class BaseBuyerAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()
    @abstractmethod
    def define_personality(self) -> Dict[str, Any]: pass
    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]: pass
    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]: pass
    @abstractmethod
    def get_personality_prompt(self) -> str: pass

# ===============================
# YOUR AGENT IMPLEMENTATION
# ===============================
class YourBuyerAgent(BaseBuyerAgent):
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "data-analyst",
            "traits": ["logical", "strategic", "budget-conscious", "calm"],
            "negotiation_style": (
                "Starts with a logical offer based on data. Avoids emotion, uses comparisons. "
                "Will walk if the numbers don’t work."
            ),
            "catchphrases": [
                "Let's break this down.",
                "I'm here for value, not fluff.",
                "Let the numbers speak."
            ]
        }
    def get_personality_prompt(self) -> str:
        return LLM_SYSTEM_PROMPT

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        round_num = context.current_round or 1
        if USE_LLM:
            prompt = (
                f"You are negotiating for {context.product.quantity} boxes of {context.product.name}.\n"
                f"Your budget: ₹{context.your_budget}. Market price: ₹{context.product.base_market_price}.\n"
                f"Round: {round_num}/10.\n"
                "For your first offer, ALWAYS start at least 25% below market price, and never above 80% of market price. "
                "Your goal is to maximize savings, moving your offer upward only slightly with each new round. "
                "Give your offer and a message (stay professional and data-driven).\n"
                "Respond ONLY in JSON: { \"offer\": <number>, \"message\": \"Your message here\" }"
            )
            llm_response = call_llm(prompt)
            offer, msg = extract_offer_from_json(llm_response, context.your_budget, fallback=int(context.product.base_market_price*0.65))
            offer = clamp_offer(offer, context, round_num)
            return offer, msg
        else:
            offer = int(context.product.base_market_price * 0.65)
            offer = min(offer, context.your_budget)
            msg = f"My initial offer is ₹{offer}. {random.choice(self.personality['catchphrases'])}"
            return offer, msg

    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        round_num = context.current_round
        if not USE_LLM:
            counter_offer = max(int(seller_price * 0.85), int(context.your_budget * 0.88))
            return DealStatus.ONGOING, counter_offer, f"Fallback offer: ₹{counter_offer}"

        convo_history = "\n".join([f"{m['role'].capitalize()}: {m['message']}" for m in context.messages[-2:]])
        prompt = (
            f"You are negotiating for {context.product.quantity} boxes of {context.product.name}.\n"
            f"Your budget: ₹{context.your_budget}. Seller's offer: ₹{seller_price}.\n"
            f"Round: {round_num}/10. Market price: ₹{context.product.base_market_price}.\n"
            f"Latest seller message: \"{seller_message}\"\n"
            "Your strategy: increase your offer only in small increments, do not accept unless the price is at least 15% below market, "
            "or it's the final two rounds. Always try to get a better price by counter-offering.\n"
            "Respond ONLY in JSON: { \"offer\": <number>, \"message\": \"Your message here\" }"
        )
        llm_response = call_llm(prompt)
        offer, message = extract_offer_from_json(llm_response, context.your_budget, fallback=seller_price)
        offer = clamp_offer(offer, context, round_num)
        # Accept if offer >= seller or message signals acceptance in late rounds
        if (offer >= seller_price and round_num >= 6) or ("accept" in message.lower() and round_num >= 6):
            return DealStatus.ACCEPTED, seller_price, message
        return DealStatus.ONGOING, offer, message

# ===============================
# STOCHASTIC (NATURAL) SELLER
# ===============================
class MockSellerAgent:
    def __init__(self, min_price: int):
        self.min_price = min_price
    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        price = int(product.base_market_price * random.uniform(1.45, 1.55))  # slight randomness
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price}."
    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        # Stochastic deal trigger
        accept_factor = random.uniform(1.02, 1.13)
        if buyer_offer >= self.min_price * accept_factor:
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
        if round_num >= 9:
            counter = max(self.min_price, int(buyer_offer * random.uniform(1.01, 1.07)))
            return counter, f"Final offer: ₹{counter}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * random.uniform(1.07, 1.15)))
            return counter, f"I can come down to ₹{counter}.", False

# ===============================
# TEST HARNESS
# ===============================
def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int) -> Dict[str, Any]:
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(product, buyer_budget, 0, [], [], [])
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    start_time = time.time()
    for round_num in range(10):
        context.current_round = round_num + 1
        if round_num == 0:
            offer, msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, offer, msg = buyer_agent.respond_to_seller_offer(context, seller_price, seller_msg)
        context.your_offers.append(offer)
        context.messages.append({"role": "buyer", "message": msg})
        if status == DealStatus.ACCEPTED:
            return {
                "deal_made": True,
                "final_price": seller_price,
                "rounds": context.current_round,
                "savings": buyer_budget - seller_price,
                "conversation": context.messages,
                "time_taken": round(time.time() - start_time, 2),
                "llm_used": USE_LLM
            }
        seller_price, seller_msg, accepted = seller.respond_to_buyer(offer, round_num)
        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})
        if accepted:
            return {
                "deal_made": True,
                "final_price": offer,
                "rounds": context.current_round,
                "savings": buyer_budget - offer,
                "conversation": context.messages,
                "time_taken": round(time.time() - start_time, 2),
                "llm_used": USE_LLM
            }
    return {
        "deal_made": False,
        "final_price": None,
        "rounds": 10,
        "savings": 0,
        "conversation": context.messages,
        "time_taken": round(time.time() - start_time, 2),
        "llm_used": USE_LLM
    }

# ===============================
# 3-SCENARIO TEST RUNNER
# ===============================
def test_real_scenarios():
    print("\n========================")
    print("TESTING REAL SCENARIOS")
    print("========================")
    agent = YourBuyerAgent("SmartBuyer")
    scenarios = [
        {
            "name": "Easy Market",
            "product": Product(
                name="Alphonso Mangoes",
                category="Mangoes",
                quantity=100,
                quality_grade="A",
                origin="Ratnagiri",
                base_market_price=180000,
                attributes={}
            ),
            "budget": 200000,
            "seller_min": 150000
        },
        {
            "name": "Tight Budget",
            "product": Product(
                name="Kesar Mangoes",
                category="Mangoes",
                quantity=150,
                quality_grade="B",
                origin="Gujarat",
                base_market_price=150000,
                attributes={}
            ),
            "budget": 140000,
            "seller_min": 125000
        },
        {
            "name": "Premium Product",
            "product": Product(
                name="Export-Grade Mangoes",
                category="Mangoes",
                quantity=50,
                quality_grade="Export",
                origin="Devgad",
                base_market_price=200000,
                attributes={}
            ),
            "budget": 190000,
            "seller_min": 175000
        }
    ]
    deals = 0
    total_savings = 0
    all_results = []
    for s in scenarios:
        print(f"\nRunning Scenario: {s['name']}")
        print(f"Product: {s['product'].name} | Budget: ₹{s['budget']} | Market Price: ₹{s['product'].base_market_price}")
        print(f"LLM Mode: {'Enabled' if USE_LLM else 'Disabled'}")
        print("Max Rounds: 10")
        result = run_negotiation_test(agent, s["product"], s["budget"], s["seller_min"])
        all_results.append((s['name'], result))
        if result['deal_made']:
            deals += 1
            total_savings += result['savings']
        # Print conversation
        print("--- Conversation ---")
        for msg in result['conversation']:
            print(f"{msg['role'].capitalize()}: {msg['message']}")
        if result['deal_made']:
            print(f"DEAL at ₹{result['final_price']} in {result['rounds']} rounds | Savings: ₹{result['savings']} | Time: {result['time_taken']}s")
        else:
            print(f"NO DEAL after 10 rounds")
    print("\n========================")
    print("FINAL RESULTS SUMMARY")
    print("========================")
    for scenario_name, result in all_results:
        print(f"{scenario_name}")
        print(f"   - Deal: {'Yes' if result['deal_made'] else 'No'}")
        print(f"   - Rounds: {result['rounds']}")
        print(f"   - Savings: ₹{result['savings']}")
        print(f"   - Time Taken: {result['time_taken']}s")
        print(f"   - LLM Used: {'Yes' if result['llm_used'] else 'No'}")
    print(f"\nTotal Savings: ₹{total_savings}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM usage")
    args = parser.parse_args()
    if args.no_llm:
        USE_LLM = False
    test_real_scenarios()
