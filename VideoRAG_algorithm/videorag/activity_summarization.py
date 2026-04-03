import asyncio
from collections import defaultdict
import json
from .prompt import GRAPH_FIELD_SEP

async def extract_activities(clip_text: str, llm_func: callable) -> list[str]:
    """
    Extracts a list of activities from the given clip text using the LLM.
    """
    prompt = (
        "You are an AI assistant tasked with extracting activities from video clip descriptions. "
        "Read the following text and list the distinct activities taking place. "
        "Focus on actions and events. "
        "Return ONLY the activities as a comma-separated list, or 'None' if no recognizable activities are present.\n"
        f"Text:\n{clip_text}\n\nActivities:"
    )
    result = await llm_func(prompt)
    if not result or "none" in result.lower().strip() or not result.strip():
        return []
    # parse comma-separated list
    activities = [act.strip().lower() for act in result.split(",") if act.strip()]
    return list(set(activities))

async def group_by_activity(knowledge_graph_inst, text_chunks_db=None, target_video_name: str = None) -> dict[str, list[str]]:
    """
    Groups nodes in the graph by their associated activities.
    """
    activity_clusters = defaultdict(list)
    # NetworkXStorage provides _graph.nodes(data=True)
    if not hasattr(knowledge_graph_inst, "_graph"):
        return {}
        
    for node_id, node_data in knowledge_graph_inst._graph.nodes(data=True):
        if target_video_name and text_chunks_db and "source_id" in node_data:
            source_list = node_data["source_id"].split(GRAPH_FIELD_SEP)
            pieces = await text_chunks_db.get_by_ids(source_list)
            belongs_to_video = False
            for p in pieces:
                if p and "video_segment_id" in p:
                    for vs_id in p["video_segment_id"]:
                        if target_video_name in vs_id:
                            belongs_to_video = True
                            break
                if belongs_to_video:
                    break
            if not belongs_to_video:
                continue

        if "activities" in node_data:
            try:
                acts = json.loads(node_data["activities"])
                for act in acts:
                    if act:
                        activity_clusters[act].append(node_id)
            except Exception:
                pass
    return dict(activity_clusters)

async def traverse_subgraph(knowledge_graph_inst, nodes: list[str], top_k: int = 5, depth: int = 2) -> list[str]:
    """
    Select representative nodes by degree centrality, then traverse BFS up to `depth`.
    """
    # 1. Select representative nodes
    node_degrees = []
    for n in nodes:
        deg = await knowledge_graph_inst.node_degree(n)
        node_degrees.append((n, deg))
    
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    representatives = [n[0] for n in node_degrees[:top_k]]
    
    # 2. BFS
    from collections import deque
    reachable = set()
    queue = deque([(r, 0) for r in representatives])
    
    while queue:
        curr_node, curr_depth = queue.popleft()
        if curr_node not in reachable:
            reachable.add(curr_node)
            if curr_depth < depth:
                edges = await knowledge_graph_inst.get_node_edges(curr_node)
                if edges:
                    for src, tgt in edges:
                        # edges can be (src, tgt) depending on Graph type
                        neighbor = tgt if src == curr_node else src
                        if neighbor not in reachable:
                            queue.append((neighbor, curr_depth + 1))
                            
    return list(reachable)

async def extract_text(nodes: list[str], knowledge_graph_inst, text_chunks_db, target_video_name: str = None) -> str:
    """
    Extracts text from the given nodes by looking up their source_id chunks.
    """
    chunk_ids = set()
    for n in nodes:
        node_data = await knowledge_graph_inst.get_node(n)
        if node_data and "source_id" in node_data:
            source_list = node_data["source_id"].split(GRAPH_FIELD_SEP)
            for s in source_list:
                chunk_ids.add(s)
                
    chunk_texts = []
    if chunk_ids:
        pieces = await text_chunks_db.get_by_ids(list(chunk_ids))
        for p in pieces:
            if p and "content" in p:
                if target_video_name:
                    belongs_to_video = False
                    if "video_segment_id" in p:
                        for vs_id in p["video_segment_id"]:
                            if target_video_name in vs_id:
                                belongs_to_video = True
                                break
                    if not belongs_to_video:
                        continue
                chunk_texts.append(p["content"])
            
    return "\n\n".join(chunk_texts)

async def summarize_text(text: str, activity: str, llm_func: callable) -> str:
    """
    Local summarization for an activity cluster.
    """
    prompt = (
        f"Summarize the following text focusing on this activity: {activity}. "
        "Include what happens, key entities, and sequence.\n\n"
        f"Text:\n{text}\n\nSummary:"
    )
    summary = await llm_func(prompt)
    return summary

async def generate_activity_summary(knowledge_graph_inst, text_chunks_db, global_config: dict, target_video_name: str = None) -> str:
    """
    Orchestrator function that combines everything into a final global summary.
    """
    use_llm_func = global_config["llm"]["best_model_func"]
    
    print(f"Grouping by activity for video: {target_video_name if target_video_name else 'ALL'}...")
    activity_clusters = await group_by_activity(knowledge_graph_inst, text_chunks_db, target_video_name)
    if not activity_clusters:
        return "No activities found in the graph."
        
    local_summaries = []
    
    # Process each activity
    # limit to top N activities if there are too many, but let's process all for now
    for activity, nodes in activity_clusters.items():
        print(f"Processing activity '{activity}' with {len(nodes)} nodes...")
        reachable_nodes = await traverse_subgraph(knowledge_graph_inst, nodes, top_k=3, depth=2)
        print(f"  Reachable nodes: {len(reachable_nodes)}")
        
        extracted_text = await extract_text(reachable_nodes, knowledge_graph_inst, text_chunks_db, target_video_name)
        if not extracted_text.strip():
            continue
            
        print(f"  Summarizing text for activity '{activity}'...")
        local_summary = await summarize_text(extracted_text, activity, use_llm_func)
        local_summaries.append(f"Activity: {activity}\nSummary: {local_summary}")
        
    if not local_summaries:
        return "Not enough data to summarize."
        
    print("Generating global summary...")
    combined_text = "\n\n---\n\n".join(local_summaries)
    global_prompt = (
        "Combine these activity summaries into a coherent structured summary. "
        "Remove redundancy and format the output clearly focusing on the main events.\n\n"
        f"{combined_text}\n\nGlobal Summary:"
    )
    
    global_summary = await use_llm_func(global_prompt)
    return global_summary
