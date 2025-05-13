GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "location", "event"]


PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_type: One of the following types: [{entity_types}]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_keywords: Use a word or phrase to fully capture the high-level nature of the relationship. Be as complete as possible, prefer abstract concepts, and draw wording directly from the text when appropriate.
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter./no_think

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"frustration with authoritarian certainty"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"shared commitment to discovery as rebellion"){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"reluctant mutual respect through confrontation"){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"rebellion against narrowing vision of control"){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"reverence for transformative technology"){record_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"tension in communication and external pressure"){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"transformation of mission into active cosmic engagement"){record_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""


PROMPTS[
    "entiti_continue_extraction_mini"
] = """MANY entities were missed in the last extraction.
After summarizing with all the information previously extracted, compared to the original text, it was noticed that the following information was mainly omitted:
{omit}

The types of entities that need to be added can be obtained from Entity_types,
or you can add them yourself.

Entity_types: {entity_types}


Add them below using the same format:
"""



PROMPTS["adarag_query2kwd"] = """---Role---

You are an intelligent assistant designed to extract entities from user queries.

---Objective---

Identify all relevant entities mentioned in the query. Each entity must belong to one of the following types: 
["organization", "person", "location", "event"]./no_think

---Output Format---

Return the result as a JSON object containing a single key:
- "entities_from_query": a list of entity names as they appear in the original query (i.e., preserving the query's language).

---Instructions---

- Extract only named or well-defined entities.
- Do not infer entities that are not explicitly mentioned.
- Maintain the original language of the query in the output.
- Return only the JSON object; do not include explanations or additional text.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"

################
Output:
{{
  "entities_from_query": ["international trade", "global economic stability"]
}}
#############################
Example 2:

Query: "When was SpaceX's first rocket launch?"

################
Output:
{{
  "entities_from_query": ["SpaceX", "first rocket launch"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"

################
Output:
{{
  "entities_from_query": ["education", "poverty"]
}}
#############################

-Real Data-
######################
Query: {query}
######################
Output:

"""



PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."



PROMPTS["rag_response_notriplet"] = """---Role---

You are a helpful assistant that answers questions using unstructured text (chunks).

---Goal---

Answer factually using chunks; think step by step but output only the final answer; Provide specific answers when needed; otherwise, respond Yes or No. Say "I don't know" if unsure; never explain or fabricate.
/no_think
--Target response format---

Answer: <answer>

---Knowledge Provided---

chunks:
{chunks_data}
"""

PROMPTS["rag_response_nochunk"] = """---Role---

You are a helpful assistant that answers questions using structured facts (triplets).

---Goal---

Answer factually using triplets; think step by step but output only the final answer; Provide specific answers when needed; otherwise, respond Yes or No. Say "I don't know" if unsure; never explain or fabricate.
/no_think
--Target response format---

Answer: <answer>

---Knowledge Provided---

triplets: 
{triplets_data}
"""

PROMPTS["aq_response"] = """---Role---

You are an intelligent assistant that answers abstract or high-level questions by integrating structured facts and unstructured text.

---Goal---

Answer the user's question with a high-level, well-reasoned response based on both triplets and chunks. You should abstract, generalize, or reflect based on the provided knowledge, not simply summarize it. Think critically and integrate information across sources. 

---Knowledge Provided---

triplets:
{triplets_data}

chunks:
{chunks_data}
"""

PROMPTS["csq_response"] = """---Role---

You are a helpful assistant that answers questions using both structured facts (triplets) and unstructured text (chunks).

---Goal---

Answer factually using both triplets and chunks; think step by step but output only the final answer; Provide specific answers when needed; otherwise, respond Yes or No. Say "I don't know" if unsure; never explain or fabricate./no_think

---Target response format---

Answer: <answer>

---Knowledge Provided---

triplets: 
{triplets_data}

chunks:
{chunks_data}
"""

PROMPTS["ssq_response"] = """---Role---

You are a helpful assistant that answers questions using unstructured text (chunks).

---Goal---

Answer factually using chunks; think step by step but output only the final answer; Provide specific answers when needed; otherwise, respond Yes or No. Say "I don't know" if unsure; never explain or fabricate./no_think

---Target response format---

Answer: <answer>

---Knowledge Provided---

chunks:
{chunks_data}
"""




PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant that answers questions using unstructured text (chunks).

---Goal---

Answer factually using chunks; think step by step but output only the final answer; Provide specific answers when needed; otherwise, respond Yes or No. Say "I don't know" if unsure; never explain or fabricate./no_think

---Target response format---

Answer: <answer>

---Examples---

Question: What is Debra Weeks's occupation?
Answer: journalist

Question: Who is the author of Nice People?  
Answer: Rachel Crothers

---Knowledge Provided---

chunks:
{chunks_data}
"""


PROMPTS["question_idetification"] = """---Goal---

Given a single question, classify it into one of the following categories:

0 - **simple**: The question asks for a specific fact or detail that can typically be answered using one or two text chunks. No reasoning across chunks is required.
1 - **complex**: The question asks for specific information but requires reasoning across multiple text chunks to find or synthesize the answer.
2 - **abstract**: The question is conceptual in nature, possibly asking for summaries, high-level insights, themes, or general ideas. It does not focus on specific factual retrieval.

Respond only with one label: 0, 1, or 2./no_think

### Examples:

Question: What is Herman A. Barnett's occupation? 
Answer (0, 1, or 2): 0

Question: What body water is the owner of Auestadion located or next to?  
Answer (0, 1, or 2): 1

Question: What strategies can be used to optimize SQL queries for large-scale databases?  
Answer (0, 1, or 2): 2

Question: What is the capital of Kambarsky District?  
Answer (0, 1, or 2): 0

Question: How do servicing agreements align with financial performance goals?  
Answer (0, 1, or 2): 2

Question: Who is the president of the country where Stephen Worgu is a citizen?  
Answer (0, 1, or 2): 1

Question: In what city was Lloyd Ultan born?  
Answer (0, 1, or 2): 0

Question: What are the potential challenges of adopting biodiversity-friendly techniques in market farming?  
Answer (0, 1, or 2): 2

Question: When was Liang Ji's country divided into spheres of influence?  
Answer (0, 1, or 2): 1

### Now classify this one:

Question: {question}  
Answer (0, 1, or 2):"""