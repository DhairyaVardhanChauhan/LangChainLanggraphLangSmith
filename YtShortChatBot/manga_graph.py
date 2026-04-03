from typing import TypedDict, List
import os
import asyncio
import base64
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

llm        = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)
llm_vision = ChatMistralAI(model="pixtral-12b-2409",   temperature=0.2)

MANGADEX_API = "https://api.mangadex.org"


# ==============================
# State
# ==============================

class MangaState(TypedDict):
    manga_title:      str
    output_dir:       str
    manga_id:         str
    manga_synopsis:   str
    chapter_id:       str
    chapter_label:    str
    character_info:   str         # Tavily character roster
    story_so_far:     str         # rolling narrative accumulator
    page_paths:       List[str]   # downloaded manga page images
    page_contents:    List[str]   # vision-extracted text + scene per page
    page_scripts:     List[str]   # narration script per page
    page_audio_paths: List[str]   # one audio file per page
    video_path:       str


# ==============================
# Node 0 — Setup Output Folder
# ==============================

def setup_dirs(state: MangaState) -> MangaState:
    existing = [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and d.startswith("manga") and d[5:].isdigit()
    ]
    num = len(existing) + 1
    output_dir = os.path.join(BASE_DIR, f"manga{num}")
    os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "video_complete"), exist_ok=True)
    print(f"\n📁 Output folder: manga{num}/")
    return {"output_dir": output_dir}


# ==============================
# Node 1 — Search MangaDex
# ==============================

def search_manga(state: MangaState) -> MangaState:
    import requests

    title = state["manga_title"]
    print(f"\n🔍 Searching MangaDex for: '{title}'...")

    resp = requests.get(
        f"{MANGADEX_API}/manga",
        params={
            "title": title,
            "limit": 5,
            "availableTranslatedLanguage[]": "en",
            "contentRating[]": ["safe", "suggestive"],
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError(f"No manga found for: '{title}'")

    manga = data[0]
    manga_id = manga["id"]
    attrs = manga["attributes"]
    en_title = (
        attrs.get("title", {}).get("en")
        or attrs.get("title", {}).get("ja-ro")
        or list(attrs["title"].values())[0]
    )
    synopsis = (
        attrs.get("description", {}).get("en", "")
        or list(attrs.get("description", {}).values() or [""])[0]
    )[:600]

    print(f"   ✅ Found: {en_title}")
    return {"manga_id": manga_id, "manga_synopsis": synopsis}


# ==============================
# Node 1b — Fetch Character Info (Tavily)
# ==============================

def fetch_character_info(state: MangaState) -> MangaState:
    """Search the web for character names and roles so the LLM can name them correctly."""
    from tavily import TavilyClient

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_key:
        print("   ⚠️  TAVILY_API_KEY not set — skipping character lookup")
        return {"character_info": ""}

    title = state["manga_title"]
    print(f"\n🧑‍🤝‍🧑 Fetching character info for '{title}'...")

    client = TavilyClient(api_key=tavily_key)
    try:
        results = client.search(
            query=f"{title} main characters names roles powers",
            search_depth="basic",
            max_results=5,
        )
        snippets = [
            f"- {r['title']}: {r['content'][:300]}"
            for r in results.get("results", [])[:4]
        ]
        character_info = "\n".join(snippets)
    except Exception as e:
        print(f"   ⚠️  Character search failed: {e}")
        character_info = ""

    print(f"   ✅ Character info ready ({len(character_info)} chars)")
    return {"character_info": character_info}


# ==============================
# Node 2 — Fetch Chapter
# ==============================

def fetch_chapter(state: MangaState) -> MangaState:
    import requests

    print(f"\n📚 Fetching first English chapter...")
    resp = requests.get(
        f"{MANGADEX_API}/manga/{state['manga_id']}/feed",
        params={"translatedLanguage[]": "en", "order[chapter]": "asc", "limit": 1},
        timeout=20,
    )
    resp.raise_for_status()
    chapters = resp.json().get("data", [])
    if not chapters:
        raise ValueError("No English chapters found.")

    chapter = chapters[0]
    attrs = chapter["attributes"]
    num = attrs.get("chapter") or "1"
    title = attrs.get("title") or ""
    label = f"Chapter {num}" + (f": {title}" if title else "")
    print(f"   ✅ {label}")
    return {"chapter_id": chapter["id"], "chapter_label": label}


# ==============================
# Node 3 — Download Pages
# ==============================

def download_pages(state: MangaState) -> MangaState:
    import requests

    assets_dir = os.path.join(state["output_dir"], "assets")
    print(f"\n🖼️  Downloading manga pages...")

    resp = requests.get(f"{MANGADEX_API}/at-home/server/{state['chapter_id']}", timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    server_url   = payload["baseUrl"]
    chapter_hash = payload["chapter"]["hash"]
    filenames    = (payload["chapter"].get("dataSaver") or payload["chapter"]["data"])[:10]
    quality      = "data-saver" if payload["chapter"].get("dataSaver") else "data"

    page_paths = []
    for i, fn in enumerate(filenames):
        url  = f"{server_url}/{quality}/{chapter_hash}/{fn}"
        path = os.path.join(assets_dir, f"page_{i:02d}.jpg")
        print(f"   🌐 URL: {url}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            page_paths.append(path)
            print(f"   ✅ Page {i + 1}/{len(filenames)}")
        else:
            print(f"   ⚠️  Page {i + 1} skipped ({r.status_code})")
        time.sleep(0.3)

    return {"page_paths": page_paths}


# ==============================
# Node 4 — Extract Text per Page (Groq Vision)
# ==============================

def extract_page_content(state: MangaState) -> MangaState:
    """Use Groq vision to read dialogue + describe scene for each page."""
    from PIL import Image
    import io

    print(f"\n👁️  Reading manga pages with vision model...")
    page_contents = []

    for i, path in enumerate(state["page_paths"]):
        # Resize to max 1024px before sending to vision API
        img = Image.open(path).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        char_ctx = (
            f"\n\nKnown characters in this manga:\n{state['character_info']}"
            if state.get("character_info") else ""
        )
        text_prompt = (
            "This is a manga page."
            + char_ctx
            + "\n\nExtract and structure the following:\n"
            "1. DIALOGUE: All spoken text in reading order. "
            "If you can identify a character by name, prefix their line with their name (e.g. 'Goku: ...').\n"
            "2. SCENE: One sentence — who is present and what physical action is happening.\n\n"
            "IMPORTANT — Do NOT mention or describe any of the following:\n"
            "- Foreign language characters, Chinese/Japanese/Korean text, or untranslated scripts\n"
            "- Speech bubble shapes, panel borders, or page layout\n"
            "- Image quality, scan artifacts, or visual formatting\n"
            "- The fact that this is a manga, comic, or drawn image\n\n"
            "Format your response exactly as:\n"
            "DIALOGUE: <dialogue here, Name: line format if known>\n"
            "SCENE: <one sentence: who is here, what are they doing>"
        )
        msg = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
            {"type": "text", "text": text_prompt},
        ])

        try:
            response = llm_vision.invoke([msg])
            content = response.content.strip()
        except Exception as e:
            content = f"DIALOGUE: (unreadable)\nSCENE: Page {i + 1} of the manga."
            print(f"   ⚠️  Vision failed for page {i + 1}: {e}")

        page_contents.append(content)
        print(f"   📖 Page {i + 1}/{len(state['page_paths'])} read")
        time.sleep(0.5)  # Groq rate limit

    return {"page_contents": page_contents}


# ==============================
# Node 5 — Generate Per-Page Narration Scripts
# ==============================

def generate_page_scripts(state: MangaState) -> MangaState:
    """Generate story narration per page using rolling context so each page continues from the last."""
    print(f"\n✍️  Writing story narration for each page...")

    manga_ctx      = f"{state['manga_title']} — {state['chapter_label']}"
    synopsis_block = f"Series synopsis: {state['manga_synopsis']}" if state.get("manga_synopsis") else ""
    char_block     = f"Key characters:\n{state['character_info']}"  if state.get("character_info")  else ""
    page_scripts   = []
    story_so_far   = ""  # rolling accumulator — grows each iteration

    for i, content in enumerate(state["page_contents"]):
        story_block = (
            f"Story so far (previous pages):\n{story_so_far.strip()}"
            if story_so_far.strip() else "This is the opening page."
        )

        prompt = f"""You are a skilled manga storyteller narrating "{manga_ctx}" as a YouTube Short voiceover.

{synopsis_block}
{char_block}

{story_block}

Now, page {i + 1} content extracted from the image:
{content}

Your task: Write exactly 25–40 words of spoken narration continuing the story.

Rules:
- You are a narrator TELLING THE STORY, not describing an image
- Use character names when known — never say "a character" if you know the name
- Present tense, energetic, cinematic tone
- Weave dialogue naturally — do not quote it verbatim
- NEVER say: "on this page", "in this panel", "the image shows", "we see", "the manga shows"
- NEVER mention speech bubbles, text boxes, panels, or visual elements
- 25–40 words exactly

Output ONLY the narration. No labels, no preamble."""

        response = llm.invoke([HumanMessage(content=prompt)])
        script = response.content.strip()
        page_scripts.append(script)
        story_so_far += f"\nPage {i + 1}: {script}"
        print(f"   📝 Page {i + 1}: {script[:70]}...")

    return {"page_scripts": page_scripts, "story_so_far": story_so_far}


# ==============================
# Node 5b — Refine Scripts (batch pass)
# ==============================

def refine_scripts(state: MangaState) -> MangaState:
    """Single LLM call that reviews all scripts together and fixes flow, names, and literal descriptions."""
    print(f"\n✨ Refining scripts as a complete narrative...")

    manga_ctx  = f"{state['manga_title']} — {state['chapter_label']}"
    char_block = f"Key characters:\n{state['character_info']}" if state.get("character_info") else ""
    n          = len(state["page_scripts"])

    numbered = "\n".join(
        f"Page {i + 1}: {script}"
        for i, script in enumerate(state["page_scripts"])
    )

    prompt = f"""You are a senior script editor for a manga YouTube Short: "{manga_ctx}".
{char_block}

Below are the per-page narration scripts. Review them as a COMPLETE narrative and return an improved version.

CURRENT SCRIPTS:
{numbered}

Your editing tasks:
1. Remove any remaining image descriptions — rewrite anything that says "we see", "the image shows", "there are speech bubbles", "a figure", "a character" (when the name is known), or similar meta-language as pure story narration
2. Use character names consistently — if a character is named anywhere, use the same name throughout
3. Ensure each page flows naturally from the previous one — no jarring jumps
4. Enforce 25–40 words per page — trim or expand as needed
5. Keep present tense, energetic, cinematic tone throughout

Return ONLY the refined scripts in this exact format — one per line:
Page 1: <refined narration>
Page 2: <refined narration>
...continue for all {n} pages."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    refined = []
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("page ") and ":" in line:
            _, _, text = line.partition(":")
            refined.append(text.strip())

    if len(refined) != n:
        print(f"   ⚠️  Parse mismatch ({len(refined)} vs {n}) — keeping originals")
        refined = state["page_scripts"]
    else:
        print(f"   ✅ {n} scripts refined successfully")

    for i, s in enumerate(refined):
        print(f"   📝 Page {i + 1}: {s[:70]}...")

    return {"page_scripts": refined}


# ==============================
# Node 6 — Generate Per-Page Voiceovers
# ==============================

def generate_page_voiceovers(state: MangaState) -> MangaState:
    """Generate one audio file per page. Page duration = audio duration."""
    assets_dir = os.path.join(state["output_dir"], "assets")
    print(f"\n🎙️  Generating per-page voiceovers...")

    # Load Kokoro once
    kokoro = None
    try:
        from kokoro_onnx import Kokoro
        model_path  = os.path.join(BASE_DIR, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(BASE_DIR, "voices-v1.0.bin")
        kokoro = Kokoro(model_path, voices_path)
        print("   🔊 Using Kokoro TTS")
    except Exception as e:
        print(f"   ⚠️  Kokoro unavailable: {e} — using edge-tts fallback")

    page_audio_paths = []

    for i, script in enumerate(state["page_scripts"]):
        audio_path = os.path.join(assets_dir, f"page_{i:02d}_audio.wav")

        if kokoro:
            import soundfile as sf
            samples, sr = kokoro.create(script, voice="am_adam", speed=0.93, lang="en-us")
            sf.write(audio_path, samples, sr)
        else:
            import edge_tts
            audio_path = audio_path.replace(".wav", ".mp3")

            async def _gen(text, out):
                await edge_tts.Communicate(text, "en-US-GuyNeural").save(out)

            asyncio.run(_gen(script, audio_path))

        page_audio_paths.append(audio_path)
        print(f"   ✅ Page {i + 1}/{len(state['page_scripts'])} audio ready")

    return {"page_audio_paths": page_audio_paths}


# ==============================
# Node 7 — Assemble Video
# ==============================

def create_video(state: MangaState) -> MangaState:
    """
    Each manga page is shown for exactly its narration audio duration.
    Ken Burns pan/zoom keeps the image dynamic.
    Pages only change when narrator finishes.
    """
    from moviepy import (
        AudioFileClip, ImageClip, VideoClip,
        concatenate_videoclips, CompositeVideoClip,
    )
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    TARGET_W, TARGET_H = 1080, 1920
    video_path = os.path.join(state["output_dir"], "video_complete", "manga_short.mp4")
    print(f"\n🎬 Assembling video...")

    def prepare_page(path: str) -> np.ndarray:
        """Fit manga page into 9:16 canvas with blurred background."""
        img = Image.open(path).convert("RGB")
        scale = TARGET_H / img.height
        new_w = int(img.width * scale)
        if new_w >= TARGET_W:
            img = img.resize((new_w, TARGET_H), Image.LANCZOS)
            x1 = (new_w - TARGET_W) // 2
            img = img.crop((x1, 0, x1 + TARGET_W, TARGET_H))
        else:
            bg = img.resize((TARGET_W, TARGET_H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(22))
            img = img.resize((new_w, TARGET_H), Image.LANCZOS)
            bg.paste(img, ((TARGET_W - new_w) // 2, 0))
            img = bg
        return np.array(img)

    def ken_burns_clip(page_arr: np.ndarray, duration: float, index: int) -> VideoClip:
        ZOOM = 1.20
        zoom_in = index % 2 == 0
        pan_dirs = ["left", "right", "up", "down"]
        pan = pan_dirs[index % len(pan_dirs)]

        h, w = page_arr.shape[:2]
        big_w, big_h = int(w * ZOOM), int(h * ZOOM)
        big_arr = np.array(Image.fromarray(page_arr).resize((big_w, big_h), Image.LANCZOS))

        def make_frame(t):
            progress = min(t / duration, 1.0)
            scale_t = (1.0 + (ZOOM - 1.0) * progress) if zoom_in else (ZOOM - (ZOOM - 1.0) * progress)
            cw, ch = int(w * scale_t), int(h * scale_t)
            frame_img = Image.fromarray(big_arr).resize((cw, ch), Image.LANCZOS)
            mx, my = max(0, cw - w), max(0, ch - h)
            x1 = int(mx * progress) if pan == "left" else int(mx * (1 - progress)) if pan == "right" else mx // 2
            y1 = int(my * progress) if pan == "up"   else int(my * (1 - progress)) if pan == "down"  else my // 2
            x1, y1 = max(0, min(x1, cw - w)), max(0, min(y1, ch - h))
            return np.array(frame_img)[y1:y1 + h, x1:x1 + w]

        return VideoClip(make_frame, duration=duration)

    # Build one clip per page — duration = its audio length
    clips = []
    total_duration = 0.0
    page_audio_clips = []

    for i, (page_path, audio_path) in enumerate(
        zip(state["page_paths"], state["page_audio_paths"])
    ):
        audio_clip = AudioFileClip(audio_path)
        duration   = audio_clip.duration
        page_arr   = prepare_page(page_path)
        video_clip = ken_burns_clip(page_arr, duration, i).with_audio(audio_clip)
        clips.append(video_clip)
        page_audio_clips.append(audio_clip)
        total_duration += duration
        print(f"   🎞️  Page {i + 1}: {duration:.1f}s")

    print(f"   🔗 Concatenating {len(clips)} clips (total {total_duration:.1f}s)...")
    video = concatenate_videoclips(clips, method="compose")

    # Title overlay — manga name, first 3 sec
    title_text = f"{state['manga_title']} — {state['chapter_label']}"
    title_img = Image.new("RGB", (TARGET_W, 210), (0, 0, 0))
    ImageDraw.Draw(title_img).text(
        (TARGET_W // 2, 105), title_text,
        fill="white", anchor="mm", font_size=40,
    )
    title_clip = (
        ImageClip(np.array(title_img), duration=3.0)
        .with_opacity(0.82)
        .with_position(("center", "top"))
    )

    # Subscribe bar — last 3 sec
    sub_img = Image.new("RGB", (TARGET_W, 170), (200, 20, 20))
    ImageDraw.Draw(sub_img).text(
        (TARGET_W // 2, 85), "🔔 Subscribe for more manga!",
        fill="white", anchor="mm", font_size=40,
    )
    sub_clip = (
        ImageClip(np.array(sub_img), duration=3.0)
        .with_start(max(0, total_duration - 3.0))
        .with_opacity(0.88)
        .with_position(("center", "bottom"))
    )

    final = CompositeVideoClip([video, title_clip, sub_clip], size=(TARGET_W, TARGET_H))

    print(f"   🎥 Encoding (this takes 1-3 min)...")
    final.write_videofile(
        video_path, fps=24,
        codec="libx264", audio_codec="aac",
        logger="bar",
    )
    print(f"   ✅ Video saved → {video_path}")
    return {"video_path": video_path}


# ==============================
# Graph
# ==============================

graph = StateGraph(MangaState)

graph.add_node("setup_dirs",               setup_dirs)
graph.add_node("search_manga",             search_manga)
graph.add_node("fetch_character_info",     fetch_character_info)
graph.add_node("fetch_chapter",            fetch_chapter)
graph.add_node("download_pages",           download_pages)
graph.add_node("extract_page_content",     extract_page_content)
graph.add_node("generate_page_scripts",    generate_page_scripts)
graph.add_node("refine_scripts",           refine_scripts)
graph.add_node("generate_page_voiceovers", generate_page_voiceovers)
graph.add_node("create_video",             create_video)

graph.add_edge(START,                      "setup_dirs")
graph.add_edge("setup_dirs",               "search_manga")
graph.add_edge("search_manga",             "fetch_character_info")
graph.add_edge("fetch_character_info",     "fetch_chapter")
graph.add_edge("fetch_chapter",            "download_pages")
graph.add_edge("download_pages",           "extract_page_content")
graph.add_edge("extract_page_content",     "generate_page_scripts")
graph.add_edge("generate_page_scripts",    "refine_scripts")
graph.add_edge("refine_scripts",           "generate_page_voiceovers")
graph.add_edge("generate_page_voiceovers", "create_video")
graph.add_edge("create_video",             END)

manga_bot = graph.compile()


# ==============================
# Run
# ==============================

if __name__ == "__main__":
    print("🎌 Manga Explainer — YouTube Short Generator")
    print("=============================================")
    title = input("Enter manga title: ").strip()
    result = manga_bot.invoke({"manga_title": title})
    print(f"\n✅ Done! Video: {result.get('video_path')}")
