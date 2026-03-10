from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=200)

text = """Golden hues kiss the sky,Whispering secrets to the dawn.Birds awaken with a sigh,As morning breaks, night withdrawn.Silent trees sway in the breeze,A symphony of nature's ease.Sunlight dances, colors bloom,In the quiet morning's room.The world awakes to light anew,Bathed in morning's gentle dew."""

ans = splitter.split_text(text)
print(ans)
