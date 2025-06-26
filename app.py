from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional,Literal
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch




label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]


model_path = "/content/drive/MyDrive/ATG_Assignment2/finetuned_emotion_model"


peft_config = PeftConfig.from_pretrained(model_path)


base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=len(label_names)  # Should be 6
)


model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)




class GraphState(TypedDict):
    input: str
    prediction: Optional[str]
    confidence: Optional[float]
    clarified_input: Optional[str]

# Inference Node
def inference_node(state: GraphState) -> GraphState:
    print("[Inference Node ]")
    inputs = tokenizer(state["input"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        label = label_names[predicted_class.item()]
        state = {
            **state,
            "prediction": label,
            "confidence": confidence.item()
        }
        print(f"\nLow confidence: {state['confidence']*100:.2f}%")
        print(f"Model prediction: {state['prediction']}")
        return state

# Router Function — Must return STRING!
def confidence_check_router(state: GraphState) -> str:
    if state["confidence"] < 0.65:
        print("[Confidence Node Check ] : Low | Triggering Fallback.......")
        return "fallback"
    else:
        print("[Confidence Node Check ] : Good")
        return END

# Fallback Node
def fallback_node(state: GraphState) -> GraphState:
    print("[FallBack Node ]")
    correct = input("Please clarify the correct emotion : from (sadness, joy, love, anger, fear, surprise) ").strip().lower()
    return {
        **state,
        "prediction": correct,
        "clarified_input": state["input"]
    }


graph_builder = StateGraph(GraphState)

graph_builder.add_node("inference", inference_node)
graph_builder.add_node("fallback", fallback_node)

graph_builder.set_entry_point("inference")

graph_builder.add_conditional_edges(
    "inference",
    confidence_check_router,
    {
        "fallback": "fallback",
        END: END
    }
)

graph_builder.add_edge("fallback", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter text to classify (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        final_state = graph.invoke({"input": user_input})
        print(f"Final Label: {final_state['prediction']}")
        print(f"Confidence: {final_state['confidence']*100:.2f}%")

