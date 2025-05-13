import asyncio
import re
from typing import Union
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, util
import warnings
import json_repair
from sentence_transformers import CrossEncoder
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Set
import heapq

rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

from .utils import (
    split_string_by_multi_markers,
    logger,
    process_combine_contexts,
    clean_str,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS

from tqdm import tqdm
# 先使用"\n……"分割，如果单个大于固定长度再切断
def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    paragraphs = [p.strip() for p in re.split(r"\n+", content) if p.strip()]
    results = []
    for paragraph in tqdm(paragraphs, desc="Chunking paragraphs"):
      tokens = encode_string_by_tiktoken(paragraph, model_name=tiktoken_model)

      for index, start in enumerate(
          range(0, len(tokens), max_token_size - overlap_token_size)
      ):
          chunk_content = decode_tokens_by_tiktoken(
              tokens[start : start + max_token_size], model_name=tiktoken_model
          )
          results.append(
              {
                  "tokens": min(max_token_size, len(tokens) - start),
                  "content": chunk_content.strip(),
                  "chunk_order_index": index,
              }
          )
    # print(results)
    return results


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1])
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])

    edge_keywords = clean_str(record_attributes[3])
    edge_source_id = chunk_key

    return dict(
        src_id=source,
        tgt_id=target,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        # 节点已存在，直接跳过
        return {
            "entity_name": entity_name,
            "entity_type": already_node["entity_type"],
            "source_id": already_node["source_id"],
        }

    # 节点不存在，插入新的节点信息
    entity_type = nodes_data[0]["entity_type"]
    source_id = nodes_data[0]["source_id"]

    node_data = dict(
        entity_type=entity_type,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):  
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        return dict(
            src_id=src_id,
            tgt_id=tgt_id,
            keywords=already_edge.get("keywords", ""),
        )

    keywords = GRAPH_FIELD_SEP.join(sorted(set(dp["keywords"] for dp in edges_data)))
    source_id = GRAPH_FIELD_SEP.join(set(dp["source_id"] for dp in edges_data))
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "entity_type": '"UNKNOWN"',
                },
            )

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["entity_extraction"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]

    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if "</think>" in final_result:
          final_result = final_result.split("</think>", 1)[-1].strip()

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            if "</think>" in glean_result:
              glean_result = glean_result.split("</think>", 1)[-1].strip()

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if "</think>" in if_loop_result:
              if_loop_result = if_loop_result.split("</think>", 1)[-1].strip()
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )

            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )

            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] ,
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " ,
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if entity_name_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + " " + dp["src_id"]
                + " " + dp["tgt_id"],
            }
            for dp in all_relationships_data
        }

        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst



def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts

    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_entities = chunking_by_token_size(combined_entities, max_token_size=2000)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )
    combined_relationships = chunking_by_token_size(
        combined_relationships, max_token_size=2000
    )
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    combined_sources = chunking_by_token_size(combined_sources, max_token_size=2000)
    # Format the combined context
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]

    results = await chunks_vdb.query(query, top_k=10)    
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    contents = await get_chunks_by_ids_parallel(chunks_ids, text_chunks_db)
    chunk_str = "\n".join([f"{c}" for c in contents])

    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        chunks_data=chunk_str
    )
    # print(sys_prompt)
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if "</think>" in response:
      response = response.split("</think>", 1)[-1].strip()
    return response



def scorednode2chunk(input_dict, values_dict):
    for key, value_list in input_dict.items():
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    final_chunk = Counter()
    final_chunk_id = []
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()
        id_scores_list = []
        id_scores = {}
        for d in list_of_dicts:
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]

            for id in path:
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)

        for scores in id_scores_list:
            total_id_scores.update(scores)
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    for i in final_chunk.most_common(chunk_nums):
      final_chunk_id.append(i[0])
        
    return final_chunk_id


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def clean_text(text):
    # 去除标点符号、标准化空格、小写化
    text = re.sub(r'[\".,()“”\'’]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def clean_triplet(triplet):
    return ' '.join(clean_text(item) for item in triplet)

def top_k_similar_triplets(triplets, question, k=4):
    """
    给定三元组列表和问题字符串，返回语义最相似的 top-k 三元组及相似度分数。
    """
    # 清洗数据
    cleaned_triplets = [clean_triplet(triplet) for triplet in triplets]
    cleaned_question = clean_text(question)

    # 编码
    triplet_embeddings = model.encode(cleaned_triplets, convert_to_tensor=True)
    question_embedding = model.encode(cleaned_question, convert_to_tensor=True)

    # 计算相似度
    cosine_scores = util.cos_sim(question_embedding, triplet_embeddings)[0]

    # 取 Top-K
    top_results = sorted(enumerate(cosine_scores), key=lambda x: x[1], reverse=True)[:k]

    # 返回原始三元组和分数
    return [(triplets[idx], float(score)) for idx, score in top_results]


# 优化：批量化处理CrossEncoder推理
async def rerank_triplets_batch(triplets: List[Tuple[str, str, str]], originalquery: str, top_trplets,batch_size=32):
    pairs = [(originalquery, " ".join(triplet)) for triplet in triplets]
    results = []
    with ThreadPoolExecutor() as executor:
        # 批量推理
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            scores = await asyncio.to_thread(rerank_model.predict, batch_pairs)
            results.extend(zip(scores, triplets[i:i + batch_size]))
    
    # 使用heapq替代sorted，提高效率
    top_triplets = heapq.nlargest(top_trplets, results, key=lambda x: x[0])
    return [triplet for _, triplet in top_triplets]

# 优化：并行查询数据库
async def get_chunks_by_ids_parallel(chunks_ids: List[int], text_chunks_db) -> List[str]:
    # 使用asyncio并行查询
    tasks = [text_chunks_db.get_by_ids([chunk_id]) for chunk_id in chunks_ids]
    results = await asyncio.gather(*tasks)
    # 聚合返回的结果
    contents = [unit['content'] for result in results for unit in result]
    return contents

# 优化：并发执行多步骤
async def _build_mini_query_context(
    ent_from_query,
    originalquery,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    use_model_func,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    # 并发查询 ent_from_query 的节点

    # 问题分类
    question_identifier = PROMPTS["question_idetification"]
    kw_prompt = question_identifier.format(question=originalquery)
    ques_cls = await use_model_func(kw_prompt)
    if "</think>" in ques_cls:
      ques_cls = ques_cls.split("</think>", 1)[-1].strip()
    # print(ques_cls)
    prediction = ''.join(filter(str.isdigit, ques_cls))[:1]

    top_neighbors =2
    top_chunks =4
    top_entities = 4
    top_trplets = 25
    if prediction == '0':
        # 直接返回4个相似度最高的文档
        # 简单问题的处理逻辑
        results = await chunks_vdb.query(originalquery, top_k=top_chunks)
        chunks_ids = [r["id"] for r in results]
        contents = await get_chunks_by_ids_parallel(chunks_ids, text_chunks_db)
        return 1, contents,prediction
    elif prediction == '2':
        # 抽象问题的处理逻辑
        top_chunks =6
        top_entities = 6
        top_trplets = 30

    tasks = [entity_name_vdb.query(ent, top_k=top_entities) for ent in ent_from_query]
    results_nodes = await asyncio.gather(*tasks)

    for idx, ent in enumerate(ent_from_query):
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_nodes[idx]]
        nodes_from_query_list.append(results_nodes[idx])

    candidate_reasoning_path = {}


    # 构建推理路径
    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }
        candidate_reasoning_path.update(candidate_reasoning_path_new)

    # 获取邻居路径
    tasks = [knowledge_graph_inst.get_neighbors_within_k_hops(key, top_neighbors) for key in candidate_reasoning_path.keys()]
    neighbors_paths = await asyncio.gather(*tasks)

    # 填充路径
    for idx, key in enumerate(candidate_reasoning_path.keys()):
        candidate_reasoning_path[key]["Path"] = neighbors_paths[idx]
        imp_ents.append(key)

    used_pairs = set()  # 记录已出现的 (src, tgt) 或 (tgt, src)
    entities_in_triples = set()  # 记录出现在三元组中的 imp_entities
    goodedge = set()

    # 去重和构建三元组
    for entity in imp_ents:
        for src, tgt in await knowledge_graph_inst.get_node_edges(entity):
            pair_key = tuple(sorted((src, tgt)))
            if pair_key in used_pairs:
                continue

            relation_info = await knowledge_graph_inst.get_edge(src, tgt)
            relation = relation_info.get("keywords")

            triple = (src, relation, tgt)
            goodedge.add(triple)
            used_pairs.add(pair_key)

            # 记录出现的实体
            entities_in_triples.add(src)
            entities_in_triples.add(tgt)

    triplets = list(goodedge)
    top_triplets = []

    # 只有在非空时才 rerank
    if triplets:
        top_triplets = await rerank_triplets_batch(triplets, originalquery, top_trplets)

    # 并行查询 chunks_vdb 和获取文本内容
    results = await chunks_vdb.query(originalquery, top_k=top_chunks)
    chunks_ids = [r["id"] for r in results]
    contents = await get_chunks_by_ids_parallel(chunks_ids, text_chunks_db)

    return top_triplets, contents,prediction

def extract_strings_from_1_to_end(result_str: str):
  all_strings = re.findall(r"'([^']+)'", result_str)
  return all_strings[1:]  # 跳过第一个（entities_from_query）

async def adarag_query(  # adarag
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["adarag_query2kwd"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)

    if "</think>" in result:
        result = result.split("</think>", 1)[-1].strip()

    try:
        keywords_data = json_repair.loads(result)

        if isinstance(keywords_data, dict):
            # 正常情况：直接从字典取
            entities_from_query = keywords_data.get("entities_from_query", [])

        elif isinstance(keywords_data, list):
            # 是列表，查找里面有没有 dict 含有这个字段
            entities_from_query = []
            for item in keywords_data:
                if isinstance(item, dict) and "entities_from_query" in item:
                    entities_from_query = item["entities_from_query"]
                    break
            else:
                # 没找到，退而求其次用正则方式提取
                entities_from_query = extract_strings_from_1_to_end(result)

        else:
            # fallback：如果不是 dict 或 list，就当字符串处理
            entities_from_query = extract_strings_from_1_to_end(result)

    except Exception as e:
        print(f"JSON parsing error: {e}")
        entities_from_query = extract_strings_from_1_to_end(result)


    triplet,chunk,prediction= await _build_mini_query_context(
        entities_from_query,
        query,
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        embedder,
        query_param,
        use_model_func,
    )
    chunk_str = "\n".join([f"{c}" for c in chunk])

    if prediction == '0':
        sys_prompt_temp = PROMPTS["ssq_response"]
        sys_prompt = sys_prompt_temp.format(
            chunks_data=chunk_str
        )
    elif prediction == '1': 
      triplet_str = " ".join([f"{t}" for t in triplet]) 
      triplet_str =triplet_str.replace("<SEP>", "|")
      sys_prompt_temp = PROMPTS["csq_response"]
      sys_prompt = sys_prompt_temp.format(
            triplets_data=triplet_str,
            chunks_data=chunk_str
        )     
    else:    
      triplet_str = " ".join([f"{t}" for t in triplet]) 
      triplet_str =triplet_str.replace("<SEP>", "|")
      sys_prompt_temp = PROMPTS["aq_response"]
      sys_prompt = sys_prompt_temp.format(
            triplets_data=triplet_str,
            chunks_data=chunk_str
        )
    
    # print(sys_prompt)
    response = await use_model_func(
          query,
          system_prompt=sys_prompt,
    )

    if "</think>" in response:
      response = response.split("</think>", 1)[-1].strip()

    return response
