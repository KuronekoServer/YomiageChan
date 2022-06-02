import discord
import os
import subprocess
import torch
import MeCab
import soundfile as sf
import alkana
import emoji
from urllib.parse import urlparse
import unicodedata as ud
import asyncio
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import numpy as np
import pyrubberband
import re
import sys

server = []
client = discord.Client(allowed_mentions=discord.AllowedMentions.none())

def newfile(filename):
    exec_cmd("touch " + filename)
    return 0

def is_new_server(sid):
    for i in range(len(server)):
        if server[i].sid == sid:
            return i
    return -1

mypath = os.getcwd() + "/"
TOKEN = "TOKEN"

def exec_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return -1

def wakati(sentence):
    wakati = MeCab.Tagger("-Owakati")
    words = wakati.parse(sentence).split()
    return words

class ServerInfo(object):
    def __init__(self, server_id):
        self.sid = server_id
        self.prefix = "yc!"
        self.cid = 0
        self.vol = 100
        self.vctype = 1
        self.abbreviation = 50
        self.dictionaly = []
        self.is_connect = 0
        self.read_dict()

    def read_dict(self):
        dbname = mypath + "lib/" + str(self.sid) + "_" + "dict.db"
        if not os.path.exists(dbname) == True:
            newfile(dbname)
            f = open(dbname, 'w')
            f.write("dummy====================>dummy")
            f.close()
        f = open(dbname, 'r')
        dict = f.read()
        f.close()
        dict = dict.split("\n")
        for item in dict:
            self.dictionaly = np.append(self.dictionaly, item.split("====================>"))
        return 0

    def add_word(self, toconvert, converted):
        dbname = mypath + "lib/" + str(self.sid) + "_" + "dict.db"
        f = open(dbname, 'w')
        f.write(toconvert + "====================>" + converted)
        f.close()
        r = np.array(toconvert, converted)
        self.dictionaly = np.append(self.dictionaly, r)
        return 0

    def delete_word(self, toconvert):
        dbname = mypath + "lib/" + str(self.sid) + "_" + "dict.db"
        self.dictionaly = np.delete(self.dictionaly, r)
        return 0

tukuyomi = Text2Speech.from_pretrained(
    model_tag=str_or_none('kan-bayashi/tsukuyomi_full_band_vits_prosody'),
    vocoder_tag=str_or_none('none'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

jsut = Text2Speech.from_pretrained(
    model_tag=str_or_none("kan-bayashi/jsut_full_band_vits_prosody"),
    vocoder_tag=str_or_none('none'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

def check_url(url):
    flag = True
    p = urlparse(url)
    query = urllib.parse.quote_plus(p.query, safe='=&')
    url = '{}://{}{}{}{}{}{}{}{}'.format(
        p.scheme, p.netloc, p.path,
        ';' if p.params else '', p.params,
        '?' if p.query else '', query,
        '#' if p.fragment else '', p.fragment)
    try:
        f = urllib.request.urlopen(url)
        f.close()
    except urllib.request.HTTPError:
        flag = False
    return flag

def preprocess(sid, sentence):
    isnew = is_new_server(sid)
    if isnew != -1:
        if sentence[:1] == "|" and sentence[len(sentence)-1:] in "|":
            sentence.replace("|", "")
            sentence += "!slow"
    if re.search("http://", sentence) != None or re.search("https://", sentence) != None:
        if check_url(sentence.split(" ")[0]) != False:
             sentence = "URLが送信されました"
        else:
             sentence = "無効なURLが送信されました"
    else:
        if re.search("<:", sentence) != None and re.search(">", sentence) != None:
            sentence = "スタンプが添付されました"
    sentence = emoji.demojize(sentence)
    if len(sentence) >= server[isnew].abbreviation:
         sentence = sentence[:server[isnew].abbreviation]
         sentence += " 以下略"
    if len(server[isnew].dictionaly) != 0:
        for i in range(int(server[isnew].dictionaly.shape[0])):
            sentence = sentence.replace(server[isnew].dictionaly[i][1], server[isnew].dictionaly[i][2])
    alp = wakati(sentence)
    for i in range(len(alp)):
        if alkana.get_kana(alp[i]) != None:
            sentence = sentence.replace(alp[i], alkana.get_kana(alp[i]))
    return sentence

def text2wav(sentence, num):
    slow = 0
    type = server[num].vctype
    if sentence[-5:] == "!slow":
        slow = 1
        sentence.replace("!slow", "")
    with torch.no_grad():
        if type == 1:
            wav = tukuyomi(sentence)["wav"]
        if type == 2:
            wav = jsut(sentence)["wav"]
    if slow == 1:
        wav = pyrubberband.time_stretch(wav, 48000, 0.5)
    sf.write(mypath + "tts/" + sentence + ".wav", wav, 48000, format="WAV", subtype="PCM_16")
    if os.path.exists(mypath + "tts/" + sentence + ".wav") == False:
          return -1
    return mypath + "tts/" + sentence + ".wav"

@client.event

async def on_ready():
    print('------')
    print("TTSBot(Using ESPNET2) Ver2 poweredby Eurobeat-Lover at YH")
    print("Bot Name:" + client.user.name)  # Botの名前
    print("Bot ID:" + str(client.user.id))  # ID
    print("library version:" + discord.__version__)  # discord.pyのバージョン
    print('------')
    await client.change_presence(activity=discord.Game(name=f"BOT利用数：{len(client.guilds)}サーバー"))

@client.event

async def on_message(message):
    sid = message.guild.id
    if not os.path.exists("is_mkdir"):
        os.mkdir(mypath + "lib/")
        os.mkdir(mypath + "state/")
        os.mkdir(mypath + "tts/")
        newfile(mypath + "is_mkdir")
    if is_new_server(sid) == -1:
        server.append(ServerInfo(sid))
    if message.author.bot:
        return
    num = is_new_server(sid)
    if message.content == "!jn":
        if message.author.voice is None:
            await message.channel.send("あなたはボイスチャンネルに接続していません。")
            return
        server[num].is_connect = 1
        server[num].cid = message.channel.id
        await message.author.voice.channel.connect()
        embed = discord.Embed(title="読み上げちゃん",description="接続しました。",color=discord.Colour.green())
        await message.channel.send(embed=embed)
        message.guild.voice_client.play(discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(mypath + "/tts/init/0.wav"), volume=(server[num].vol / 100)))
        return
    if message.content == "!lv":
        if message.guild.voice_client is None:
            await message.author.voice.channel.connect()
            embed = discord.Embed(title="読み上げちゃん",description="接続していません。",color=discord.Colour.red())
            await message.channel.send(embed=embed)            
            return
        server[num].is_connect = 0
        server[num].cid = 0
        await message.guild.voice_client.disconnect()
        embed = discord.Embed(title="読み上げちゃん",description="切断しました。",color=discord.Colour.green())
        await message.channel.send(embed=embed)
    if re.search(server[num].prefix + "volume", message.content) != None:
        if message.content == server[num].prefix + "volume":
            embed = discord.Embed(title="音量設定",description=str(server[num].vol) + "　%",color=discord.Colour.green())
            await message.channel.send(embed=embed)
        vol = int(message.content.replace(server[num].prefix + "volume ",""))
        if vol <= 150 and vol >= 30:
            server[num].vol = vol
            embed = discord.Embed(title="音量設定",description=str(vol) + "%",color=discord.Colour.green())
            await message.channel.send(embed=embed)
        else:
            embed = discord.Embed(title="読み上げちゃん",description="音量は30-150の必要があります",color=discord.Colour.red())
            await message.channel.send(embed=embed)            
    if re.search(server[num].prefix + "add", message.content) != None:
        if message.content == server[num].prefix + "add":
            embed = discord.Embed(title="辞書登録",description="辞書に単語を追加します",color=discord.Colour.green())
            await message.channel.send(embed=embed)
        words = message.content.replace(server[num].prefix + "add ","").split(" ")
        server[num].add_word(words[0], words[1])
        embed = discord.Embed(title="辞書登録",description= words[0] + " => " + words[1] ,color=discord.Colour.green())
        await message.channel.send(embed=embed)
    if re.search(server[num].prefix + "prefix", message.content) != None:
        if message.content == server[num].prefix + "prefix":
            embed = discord.Embed(title="読み上げちゃん",description="プレフィックスを変更できます。",color=discord.Colour.green())
            await message.channel.send(embed=embed)
        words = message.content.replace(server[num].prefix + "prefix ","")
        server[num].prefix = words
        embed = discord.Embed(title="プレフィックス変更",description="プレフィックスを" + words + "に変更しました",color=discord.Colour.green())
        await message.channel.send(embed=embed)
    if re.search(server[num].prefix + "abb", message.content) != None:
        if message.content == server[num].prefix + "abb":
            embed = discord.Embed(title="略の長さ変更",description="略の長さ => " + str(server[num].abbreviation) + "",color=discord.Colour.green())
            await message.channel.send(embed=embed)
        abb = int(message.content.replace(server[num].prefix + "abb ",""))
        if abb <= 150 and abb >= 30:
            server[num].abbreviation = int(abb)
            embed = discord.Embed(title="略の長さ変更",description="略の長さを変えました。 => " + str(abb) + "",color=discord.Colour.green())
            await message.channel.send(embed=embed)
        else:
            embed = discord.Embed(title="読み上げちゃん",description="略の長さは30-150の必要があります",color=discord.Colour.red())
            await message.channel.send(embed=embed)   
    if re.search(server[num].prefix + "voice", message.content) != None:
        if message.content == server[num].prefix + "voice":
            embed = discord.Embed(title="読み上げちゃん",description="声を変更します",color=discord.Colour.green())
            await message.channel.send(embed=embed)   
        type = int(message.content.replace(server[num].prefix + "voice ",""))
        if type < 3 and type > 0:
            server[num].vctype = int(type)
        else:
            embed = discord.Embed(title="使用できる声",description="1:つくよみちゃん 2:JSUT",color=discord.Colour.green())
            await message.channel.send(embed=embed)   
    if server[num].cid == message.channel.id:
        if server[num].is_connect == 1:
            sentence = message.content
            if len(message.attachments) != 0:
                file = message.attachments[0]
                type = file.filename.split(".")[1].lower()
                if type == "webp" or type == "jpg" or type == "png" or type == "gif" or type == "bmp":
                    sentence = "画像が添付されました"
                elif type == "mp3" or type == "m4a" or type == "wav" or type == "ogg" or type == "flac" or type == "aac":
                    sentence = "音声ファイルが添付されました"
                elif type == "mov" or type == "mp4" or type == "mkv" or type == "avi":
                    sentence = "動画ファイルが添付されました"
                elif type == "txt":
                    sentence = "テキストファイルが添付されました"
                else:
                    sentence = "ファイルが添付されました"
            sents = sentence.split("\n")
            for s in sents:
                if message.mentions is not None:
                    for mention in re.findall(r"@(everyone|here|[!&]?[0-9]{17,20})", message.content):
                        if mention.replace("@", "") == "everyone":
                            s = s.replace("@everyone", "@ everyone")
                        elif mention.replace("@", "") == "here":
                            s = s.replace("@here", "@ here")
                        else:
                            userId = re.match(r"[0-9]+", str(mention.replace("!", ""))).group()
                            try:
                                u = await client.fetch_user(int(userId))
                            except:
                                sentence = sentence.replace(mention, "@" + str(userId))
                            else:
                                if userId.isdigit():
                                    s = s.replace(mention, u.name)
                    else:
                        s = discord.utils.escape_mentions(s)
                if re.search("<#", sentence) != None and re.search(">", sentence) != None:
                    cn = client.get_channel(int(s[2:].replace(">", "")))
                    s = cn.name
            sentence = preprocess(sid, s)
            filename = text2wav(sentence, num)
            message.guild.voice_client.play(discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(filename), volume=(server[num].vol / 100)))
            await client.change_presence(activity=discord.Game(name=f"{len(client.guilds)} servers | 仮復旧v2 | by Eurobeat-Lover at YH"))

client.run(TOKEN)
