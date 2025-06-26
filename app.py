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
    intent: str
    prediction: Optional[str]
    confidence: Optional[float]
    clarified_input: Optional[str]




def inference_node(state: GraphState) -> GraphState:
    print("Inference")
    inputs = tokenizer(state["input"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        label = model.config.id2label[predicted_class.item()]
        return {
            **state,
            "prediction": label,
            "confidence": confidence.item()
        }

def confidence_check_node(state: GraphState)->GraphState:
    print("Confidence Check")
    if state["confidence"] < 0.65:
        state['intent'] = "fallback"
    else:
        state['intent'] = "__end__"
    return state

def confidence_check_router(state: GraphState)->str:
    return state['intent']
    
def fallback_node(state: GraphState) -> GraphState:
    print(f"\n Low confidence: {state['confidence']*100:.2f}%")
    print(f"Model prediction: {state['prediction']}")
    corrected = input("Please clarify the correct emotion (e.g., joy, anger...): ").strip().lower()
    log_to_file(
        f"Fallback Triggered | Input: '{state['input']}' | Predicted: {state['prediction']} "
        f"({state['confidence']*100:.2f}%) → Corrected: {corrected}"
    )

    return {
        **state,
        "prediction": corrected,
        "clarified_input": state["input"]
    }



graph_builder = StateGraph(GraphState)

graph_builder.add_node("inference", inference_node)
graph_builder.add_node("fallback", fallback_node)
graph_builder.add_node("conditionalnode", confidence_check_node)
graph_builder.set_entry_point("inference")


graph_builder.add_conditional_edges(
    "conditionalnode",
    confidence_check_router,
    {
        "fallback": "fallback",
        "__end__": END
    }
)
graph_builder.add_edge("inference","conditionalnode")
graph_builder.add_edge("fallback", END)

graph = graph_builder.compile()



if __name__ == "__main__":
    print("Emotion Classifier (Self-Healing Mode Enabled)")
    while True:
        user_input = input("\nEnter text to classify (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        final_state = graph.invoke({"input": user_input})
        print(f"\nFinal Label: {final_state['prediction']}")
        print(f"Confidence: {final_state['confidence']*100:.2f}%")

