from typing import TypedDict, List
import os
import json
import asyncio
import io

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ==============================
# LLM Setup
# ==============================

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)


# ==============================
# Graph State
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class VideoState(TypedDict):
    output_dir: str            # e.g. .../video3
    news_summary: str          # top political news from Tavily
    hook: str                  # 3-second attention grabber
    script: str                # 30-sec spoken voiceover script
    title: str                 # catchy video title
    hashtags: str              # relevant hashtags
    news_photo_query: str      # Pexels photo search query based on news topic
    gameplay_clips: List[str]  # paths to downloaded gameplay MP4 clips
    news_photos: List[str]     # paths to downloaded news-related photos
    audio_path: str            # path to voiceover audio
    video_path: str            # path to final MP4
    final_output: str          # compiled text summary


# ==============================
# Node 0 — Setup Output Folder
# ==============================

def setup_dirs(state: VideoState) -> VideoState:
    """Create videoN/assets/ and videoN/video_complete/ for this run."""
    existing = [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and d.startswith("video")
        and d[5:].isdigit()
    ]
    next_num = len(existing) + 1
    output_dir = os.path.join(BASE_DIR, f"video{next_num}")

    os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "video_complete"), exist_ok=True)

    print(f"\n📁 Output folder: video{next_num}/")
    return {"output_dir": output_dir}


# ==============================
# Node 1 — Fetch Latest News (Tavily)
# ==============================

def fetch_news(state: VideoState) -> VideoState:
    """Fetch breaking political news using Tavily search."""
    from tavily import TavilyClient

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY not set in .env")

    client = TavilyClient(api_key=tavily_key)
    results = client.search(
        query="breaking political news today",
        search_depth="basic",
        max_results=5,
    )

    items = results.get("results", [])
    news_text = "\n\n".join([
        f"• {item['title']}: {item['content'][:400]}"
        for item in items[:3]
    ])

    print(f"\n📰 News fetched ({len(items)} results)")
    print(news_text[:400])
    return {"news_summary": news_text}


# ==============================
# Nodes 2-5 — Script Generation (Groq LLaMA)
# ==============================

def write_hook(state: VideoState) -> VideoState:
    prompt = f"""You are Peter Griffin from Family Guy narrating political news as a YouTube Short.
Write ONE hook sentence (max 12 words) that is funny, surprised, or outraged — exactly how Peter Griffin would react to shocking news.
Use his speech style: blunt, clueless-but-confident, sometimes randomly referencing pop culture.

News: {state['news_summary'][:600]}

Examples of Peter Griffin hooks:
- "Holy crap, you won't believe what the government just did!"
- "This is worse than that time I fought the giant chicken."
- "Lois, you need to hear this because it's friggin' insane."

Output only the hook sentence. Nothing else."""
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n🎣 Hook: {response.content}")
    return {"hook": response.content.strip()}


def write_script(state: VideoState) -> VideoState:
    prompt = f"""You are Peter Griffin from Family Guy. You're doing a YouTube Short about the latest political news.
Write a 30-second spoken script (75–90 words) IN PETER GRIFFIN'S VOICE.

Latest News: {state['news_summary'][:800]}
Hook (open with this): {state['hook']}

Peter Griffin script rules:
- Start with the hook
- Explain the news like Peter would — simple, shocked, slightly confused but acting smart
- Add 1 short funny tangent or comparison ("This is worse than...", "Reminds me of that time...")
- Summarize what actually happened in plain terms
- End with: "Anyway, follow for more news. Hehehehehe."
- 75–90 words total, conversational, energetic

Output ONLY the script. No labels, no stage directions."""
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n📜 Script:\n{response.content}")
    return {"script": response.content.strip()}


def generate_title(state: VideoState) -> VideoState:
    prompt = f"""Create a catchy, clickable YouTube Shorts title (max 60 characters) for this political news short:

{state['script']}

Output only the title. No explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n🎬 Title: {response.content}")
    return {"title": response.content.strip()}


def generate_hashtags(state: VideoState) -> VideoState:
    prompt = f"""Generate 8–10 relevant YouTube hashtags for a political news short. Mix broad and trending tags.

Script: {state['script']}

Output only hashtags on one line, space-separated. Example: #News #Politics #BreakingNews"""
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n#️⃣  Hashtags: {response.content}")
    return {"hashtags": response.content.strip()}


# ==============================
# Node 6 — Generate News Photo Query
# ==============================

def generate_search_query(state: VideoState) -> VideoState:
    """Use LLM to pick the best Pexels photo search query for news topic."""
    prompt = f"""Based on this political news script, generate a simple 2-3 word search query to find relevant news photos on a stock photo site.
Good examples: "government building", "american flag", "protest crowd", "white house", "congress"

Script: {state['script']}

Output only the search query. No explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    query = response.content.strip().strip('"').strip("'")
    print(f"\n🔍 News photo query: {query}")
    return {"news_photo_query": query}


# ==============================
# Node 7a — Fetch Gameplay Footage (Pexels Videos)
# ==============================

def fetch_footage(state: VideoState) -> VideoState:
    """Download random gameplay footage from Pexels (used as background)."""
    import requests

    pexels_key = os.getenv("PEXELS_API_KEY")
    if not pexels_key:
        raise ValueError(
            "PEXELS_API_KEY not set in .env\n"
            "Get a free key at: https://www.pexels.com/api/"
        )

    # Randomize gameplay genre each run
    import random
    gameplay_queries = [
        "video game gameplay", "gaming screen", "fps game",
        "minecraft gameplay", "mobile gaming", "esports",
    ]
    query = random.choice(gameplay_queries)
    print(f"\n🎮 Fetching gameplay footage: '{query}'...")

    def search_videos(q, orientation=None):
        params = {"query": q, "per_page": 6}
        if orientation:
            params["orientation"] = orientation
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": pexels_key},
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("videos", [])

    videos = search_videos(query, "portrait") or search_videos(query)
    if not videos:
        videos = search_videos("gaming") or search_videos("technology screen")

    assets_dir = os.path.join(state["output_dir"], "assets")
    clip_paths = []
    for i, video in enumerate(videos[:4]):
        files = sorted(video["video_files"], key=lambda x: x.get("height", 9999))
        url = files[0]["link"]
        path = os.path.join(assets_dir, f"gameplay_{i}.mp4")
        print(f"   ⬇️  Gameplay clip {i + 1}/4...")
        r = requests.get(url, stream=True, timeout=90)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        clip_paths.append(path)
        print(f"   ✅ Gameplay clip {i + 1} → {path}")

    return {"gameplay_clips": clip_paths}


# ==============================
# Node 7b — Fetch News Photos (Pexels Photos)
# ==============================

def fetch_news_photos(state: VideoState) -> VideoState:
    """Download 3 news-topic photos from Pexels (shown at intervals in video)."""
    import requests

    pexels_key = os.getenv("PEXELS_API_KEY")
    query = state["news_photo_query"]
    print(f"\n📸 Fetching news photos: '{query}'...")

    r = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": pexels_key},
        params={"query": query, "per_page": 6, "orientation": "portrait"},
        timeout=30,
    )
    r.raise_for_status()
    photos = r.json().get("photos", [])

    if not photos:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": pexels_key},
            params={"query": query, "per_page": 6},
            timeout=30,
        )
        photos = r.json().get("photos", [])

    assets_dir = os.path.join(state["output_dir"], "assets")
    photo_paths = []
    for i, photo in enumerate(photos[:3]):
        url = photo["src"].get("large") or photo["src"]["original"]
        path = os.path.join(assets_dir, f"news_photo_{i}.jpg")
        print(f"   ⬇️  News photo {i + 1}/3...")
        resp = requests.get(url, timeout=30)
        with open(path, "wb") as f:
            f.write(resp.content)
        photo_paths.append(path)
        print(f"   ✅ Photo {i + 1} → {path}")

    return {"news_photos": photo_paths}


# ==============================
# Node 8 — Generate Voiceover (edge-tts — Free Microsoft TTS)
# ==============================

def generate_voiceover(state: VideoState) -> VideoState:
    """
    Generate voiceover using Kokoro TTS (free, local, high quality).
    Falls back to edge-tts if Kokoro fails.

    Kokoro voices (American male):
      am_adam   — deep, confident narrator
      am_michael — warm, broadcast-style
    """
    print(f"\n🎙️  Generating voiceover with Kokoro TTS...")

    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf

        model_path = os.path.join(os.path.dirname(__file__), "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(os.path.dirname(__file__), "voices-v1.0.bin")
        kokoro = Kokoro(model_path, voices_path)
        samples, sample_rate = kokoro.create(
            text=state["script"],
            voice="am_adam",   # deep American male — closest free option to Peter Griffin
            speed=0.92,        # slight slowdown for dramatic news delivery
            lang="en-us",
        )
        assets_dir = os.path.join(state["output_dir"], "assets")
        audio_path = os.path.join(assets_dir, "voiceover.wav")
        sf.write(audio_path, samples, sample_rate)
        print(f"   ✅ Kokoro (am_adam) → {audio_path}")
        return {"audio_path": audio_path}

    except Exception as e:
        print(f"   ⚠️  Kokoro failed: {e}")
        print(f"   ↩️  Falling back to edge-tts (GuyNeural)...")
        import edge_tts

        assets_dir = os.path.join(state["output_dir"], "assets")
        audio_path = os.path.join(assets_dir, "voiceover.mp3")

        async def _generate():
            communicate = edge_tts.Communicate(state["script"], "en-US-GuyNeural")
            await communicate.save(audio_path)

        asyncio.run(_generate())
        print(f"   ✅ edge-tts → {audio_path}")
        return {"audio_path": audio_path}


# ==============================
# Node 9 — Assemble Video (MoviePy)
# ==============================

def _fit_to_916(clip, target_w=1080, target_h=1920):
    """Scale + center-crop any clip to 9:16 (1080x1920)."""
    w, h = clip.size
    scale = max(target_w / w, target_h / h)
    clip = clip.resized((int(w * scale), int(h * scale)))
    nw, nh = clip.size
    x1 = (nw - target_w) // 2
    y1 = (nh - target_h) // 2
    return clip.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)


def create_video(state: VideoState) -> VideoState:
    """
    Assemble YouTube Short:
      - Gameplay footage as continuous background
      - News photos cut in every ~7 seconds for 2.5 sec each
      - Title overlay (top, first 3 sec) + hook overlay (bottom, always)
      - Voiceover throughout
    """
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip,
        concatenate_videoclips, CompositeVideoClip,
    )
    from PIL import Image, ImageDraw
    import numpy as np

    TARGET_W, TARGET_H = 1080, 1920
    PHOTO_DURATION = 2.5   # seconds each news photo is shown
    PHOTO_INTERVAL = 7.0   # show a photo every N seconds
    video_path = os.path.join(state["output_dir"], "video_complete", "yt_short.mp4")
    print(f"\n🎬 Assembling video...")

    audio = AudioFileClip(state["audio_path"])
    total_duration = audio.duration

    # ── 1. Build continuous gameplay background ──────────────────────────
    gameplay_paths = state["gameplay_clips"]
    seg_dur = total_duration / len(gameplay_paths)
    gameplay_segs = []
    for path in gameplay_paths:
        clip = VideoFileClip(path)
        dur = min(clip.duration, seg_dur)
        clip = clip.subclipped(0, dur)
        if dur < seg_dur:
            clip = clip.with_duration(seg_dur)
        clip = _fit_to_916(clip, TARGET_W, TARGET_H)
        gameplay_segs.append(clip)
    background = concatenate_videoclips(gameplay_segs)

    # ── 2. Build news photo clips with start times ────────────────────────
    photo_clips = []
    photo_paths = state.get("news_photos", [])
    for i, path in enumerate(photo_paths):
        start_t = PHOTO_INTERVAL * (i + 1)
        if start_t + PHOTO_DURATION > total_duration:
            break
        img = Image.open(path).convert("RGB")
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        photo_clip = (
            ImageClip(np.array(img), duration=PHOTO_DURATION)
            .with_start(start_t)
            .with_position(("center", "center"))
        )
        photo_clips.append(photo_clip)
        print(f"   📸 News photo {i + 1} at t={start_t:.1f}s")

    # ── 3. Title overlay (top bar, first 3 sec) ───────────────────────────
    title_img = Image.new("RGB", (TARGET_W, 260), (0, 0, 0))
    ImageDraw.Draw(title_img).text(
        (TARGET_W // 2, 130), state["title"],
        fill="white", anchor="mm", font_size=50,
    )
    title_clip = (
        ImageClip(np.array(title_img), duration=3.0)
        .with_opacity(0.78)
        .with_position(("center", "top"))
    )

    # ── 4. Hook overlay (bottom bar, always visible) ──────────────────────
    hook_img = Image.new("RGB", (TARGET_W, 240), (0, 0, 0))
    ImageDraw.Draw(hook_img).text(
        (TARGET_W // 2, 120), state["hook"],
        fill=(255, 215, 0), anchor="mm", font_size=44,
    )
    hook_clip = (
        ImageClip(np.array(hook_img), duration=total_duration)
        .with_opacity(0.78)
        .with_position(("center", "bottom"))
    )

    # ── 5. Composite all layers ───────────────────────────────────────────
    layers = [background] + photo_clips + [title_clip, hook_clip]
    final = CompositeVideoClip(layers, size=(TARGET_W, TARGET_H)).with_audio(audio)

    final.write_videofile(
        video_path, fps=24,
        codec="libx264", audio_codec="aac",
        logger=None,
    )

    print(f"   ✅ Video saved → {video_path}")
    return {"video_path": video_path}


# ==============================
# Node 10 — Compile Text Output
# ==============================

def compile_output(state: VideoState) -> VideoState:
    output = f"""
========================================
🎬 YOUTUBE SHORT — POLITICAL NEWS
========================================

📰 NEWS COVERED:
{state['news_summary'][:500]}...

📌 TITLE:
{state['title']}

🎣 HOOK:
{state['hook']}

📜 SCRIPT:
{state['script']}

#️⃣  HASHTAGS:
{state['hashtags']}

🎥 VIDEO FILE:
{state.get('video_path', 'not generated')}

========================================
"""
    print(output)
    return {"final_output": output}


# ==============================
# Graph
# ==============================

graph = StateGraph(VideoState)

graph.add_node("setup_dirs", setup_dirs)
graph.add_node("fetch_news", fetch_news)
graph.add_node("write_hook", write_hook)
graph.add_node("write_script", write_script)
graph.add_node("generate_title", generate_title)
graph.add_node("generate_hashtags", generate_hashtags)
graph.add_node("generate_search_query", generate_search_query)
graph.add_node("fetch_footage", fetch_footage)
graph.add_node("fetch_news_photos", fetch_news_photos)
graph.add_node("generate_voiceover", generate_voiceover)
graph.add_node("create_video", create_video)
graph.add_node("compile_output", compile_output)

graph.add_edge(START, "setup_dirs")
graph.add_edge("setup_dirs", "fetch_news")
graph.add_edge("fetch_news", "write_hook")
graph.add_edge("write_hook", "write_script")
graph.add_edge("write_script", "generate_title")
graph.add_edge("generate_title", "generate_hashtags")
graph.add_edge("generate_hashtags", "generate_search_query")
graph.add_edge("generate_search_query", "fetch_footage")
graph.add_edge("fetch_footage", "fetch_news_photos")
graph.add_edge("fetch_news_photos", "generate_voiceover")
graph.add_edge("generate_voiceover", "create_video")
graph.add_edge("create_video", "compile_output")
graph.add_edge("compile_output", END)

yt_short_bot = graph.compile()


# ==============================
# Run
# ==============================

if __name__ == "__main__":
    print("🚀 YouTube Short — Political News Generator")
    print("=============================================")
    result = yt_short_bot.invoke({})
    print(f"\n✅ Done! Video: {result.get('video_path')}")
