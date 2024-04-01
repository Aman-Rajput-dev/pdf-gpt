css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #222429
}
.chat-message.bot {
    background-color: #1a1b1f
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 0rem;
  color: #fff;
}
.st-emotion-cache-1y4p8pa{
padding:2rem 1rem 10rem;
}
.fleex{
display:flex;
flex-direction:column;
width:100%;
}
.you{
font-weight:900;
}

'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://img.freepik.com/premium-vector/bot-icon-chatbot-icon-concept-vector-illustration_230920-1327.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="fleex">
    <div class="you">PDF-GPT</div>
    <div class="message">{{MSG}}</div>
    </div>
    
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://e7.pngegg.com/pngimages/178/595/png-clipart-user-profile-computer-icons-login-user-avatars-monochrome-black-thumbnail.png">
    </div>
    <div class="fleex">
    <div class="you">You</div>
    <div class="message">{{MSG}}</div>
    </div>
</div>
'''
