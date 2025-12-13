entity = """### Instruction
Given an image and the names of objects annotated with bounding boxes, describe each object’s visual attributes in detail — including its color, state, and shape. Then, describe the spatial or semantic relationship between the objects using the tag `<relation>`.  
The final output should be formatted in **JSON**, with the following structure:

{{
  "objects": [
    {{
      "name": "<object_name>",
      "attributes": {{
        "color": "<color_description>",
        "state": "<state_description>",
        "shape": "<shape_description>"
      }}
    }},
    ...
  ],
  "relation": "<relation_description>"
}}

### Input
Image: (seagulls on the water)
Bounding boxes:  
Two large seagulls

### Response
{{
  "objects": [
    {{
      "name": "seagulls_1",
      "attributes": {{
        "color": "white feathers with light gray wings and black wingtips",
        "state": "floating calmly on the surface of the water",
        "shape": "compact body with slightly curved posture"
      }}
    }},
    {{
      "name": "seagulls_2",
      "attributes": {{
        "color": "white and gray feathers with dark wingtips",
        "state": "spreading its wings while catching a fish in its beak",
        "shape": "large wings fully extended, dynamic curved pose"
      }}
    }}
  ],
  "relation": "the right seagull is in front of the left seagull and appears to be interacting with the water while the left one watches"
}}

### Input:
Bounding boxes:
{boxes}"""


unamb_entity = """### Instruction
Given an image and the names of objects annotated with bounding boxes, describe each object’s visual attributes in detail — including its color, state, and shape.  
The final output should be formatted in **JSON**, with the following structure:

{{
  "name": "<object_name>",
  "attributes": {{
    "color": "<color_description>",
    "state": "<state_description>",
    "shape": "<shape_description>"
  }}
}}


### Input
Image: (seagulls on the water)
Bounding boxes:  
large seagulls

### Response

{{
  "name": "seagulls_1",
  "attributes": {{
    "color": "white feathers with light gray wings and black wingtips",
    "state": "floating calmly on the surface of the water",
    "shape": "compact body with slightly curved posture"
  }}
}}

### Input:
Bounding boxes:
{boxes}"""


qa = """### Instruction
Given an image and its entity dictionary, create an **ambiguous Visual Question Answering (VQA)** example.

The question should:
- Be ambiguous — not specify which object or person is being referred to (e.g., “What is the woman wearing?” rather than “What is the woman on the left wearing?”).

The answers should include:
1. A **plausible correct** answer derived from one entity’s attributes (color, state, etc.).
2. Two **incorrect but realistic** answers (different colors, states, or unrelated descriptions).
3. One **ambiguous clarification** that uses the relation information (e.g., “You mean the woman on the left?”).

The final output should be in JSON format:
{{
  "question": "<ambiguous question>",
  "answers": {{
    "1": "<correct answer>",
    "2": "<incorrect answer>",
    "3": "<incorrect answer>",
    "4": "<ambiguous clarification>"
  }}
}}

### Input
Image: (people sitting around a wooden table)  
Entity dictionary:
{{
  "object_1": {{
    "relation": "on the left",
    "color": "blue patterned top",
    "state": "sitting at a wooden table",
    "shape": "medium build with a slightly rounded posture"
  }},
  "object_2": {{
    "relation": "on the right",
    "color": "black sweater",
    "state": "leaning slightly forward and smiling",
    "shape": "slim build with a straight posture"
  }}
}}

### Response
{{
  "question": "What is the woman wearing?",
  "answers": {{
    "1": "She seems to be wearing a blue patterned top.",
    "2": "I think she’s wearing a black sweater.",
    "3": "She’s wearing blue jeans, right?",
    "4": "You mean the woman on the left?"
  }}
}}

### Input:
Entity dictionary:
{entity_dict}
### Response"""

qa_unambiguous = """### Instruction
Given an entity dictionary containing **only one object**, create a **non-ambiguous Visual Question Answering (VQA)** example.

### Requirements

#### Question
- The question must clearly refer to the single object.
- It must NOT introduce ambiguity.
- Examples: “What is the person wearing?”, “What is the object doing?”, etc.

#### Answers
Generate four answers:
1. **Correct answer** derived from the entity’s attributes.
2. **Plausible but incorrect** answer.
3. **Visually or contextually inconsistent** incorrect answer.
4. **Another incorrect** answer.
- Do **NOT** generate any clarification (only one object).

### Output Format
Return JSON formatted as:

{{
  "question": "<unambiguous question>",
  "answers": {{
    "1": "<correct answer>",
    "2": "<incorrect answer>",
    "3": "<incorrect answer>",
    "4": "<incorrect answer>"
  }}
}}

### Input
{{
  "name": "person_1",
  "attributes": {{
    "color": "red jacket",
    "state": "standing with hands in pockets",
    "shape": "tall and slim silhouette"
  }}
}}

### Response
{{
  "question": "What is the person wearing?",
  "answers": {{
    "1": "They are wearing a red jacket.",
    "2": "They are wearing a blue hoodie.",
    "3": "They seem to be wearing a white dress.",
    "4": "They are wearing a black raincoat."
  }}
}}

### Input
{entity_dict}

### Response
"""
