import asyncio
import json
import os
import time
import pyaudio
import sys
import boto3
import sounddevice
import requests
from aioconsole import ainput  # 新增异步输入库
import threading

from concurrent.futures import ThreadPoolExecutor
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

from api_request_schema import api_request_list, get_model_ids

from zhipuai import ZhipuAI # 做rag
from dotenv import load_dotenv

import multi_model

multi_input = multi_model.upload_file()
print(multi_input[0])

load_dotenv()

# 请填写您自己的APIKey
secret_name = os.getenv("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=secret_name)

# import our fine-tunning data
from fine_tunning_data import ft_data

model_id = os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if model_id not in get_model_ids():
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

api_request = api_request_list[model_id]

voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']
voiceNameList = ['Zhiyu', 'Ivy', 'Takumi', 'Seoyeon']
voicePromptList = ['Chinese', 'English', 'Japanese', 'Korean']
voice_index = 0

config = {
    'log_level': 'none',  # One of: info, debug, none
    #'last_speech': "If you have any other questions, please don't hesitate to ask. Have a great day!",
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': voiceLanguageList[voice_index],
        'VoiceId': voiceNameList[voice_index],
        'OutputFormat': 'pcm',
    },
    'bedrock': {
        'api_request': api_request
    }
}


p = pyaudio.PyAudio()
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])
polly = boto3.client('polly', region_name=config['region'])
transcribe_streaming = TranscribeStreamingClient(region=config['region'])

def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)




class UserInputManager:
    shutdown_executor = False
    executor = None
    _stop_event = threading.Event()  # 新增静态事件标志
    _audio_controller = None  # 新增音频控制引用

    @classmethod
    def bind_audio_controller(cls, controller):
        """绑定音频控制器实例"""
        cls._audio_controller = controller

    @staticmethod
    def set_executor(executor):
        UserInputManager.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputManager.shutdown_executor = False
        raise Exception()  # Workaround to shut down exec, as executor.shutdown() doesn't work as expected.

    @staticmethod
    def start_user_input_loop():
        while True:
            user_input = sys.stdin.readline().strip()
            printer(f'[DEBUG] 收到用户输入: {user_input}', 'debug')

            # 新增语音控制命令处理
            if user_input.lower() == 'stop':
                if UserInputManager._audio_controller:
                    UserInputManager._audio_controller.stop_audio()
                else:
                    print("[警告] 音频控制器未初始化")
            else:
                UserInputManager.shutdown_executor = True

    @staticmethod
    def is_executor_set():
        return UserInputManager.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputManager.shutdown_executor

# 新增文字输入处理类
class TextInputManager:
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.running = True

    # async def start_text_input(self):
    #     """异步监听文本输入"""
    #     loop = asyncio.get_event_loop()
    #     reader = asyncio.StreamReader()
    #     protocol = asyncio.StreamReaderProtocol(reader)
    #     await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    #
    #     while self.running:
    #         try:
    #             # 使用aioconsole实现非阻塞输入
    #             text = await ainput("")
    #             await self.input_queue.put(text.strip())
    #             print(f"[文本输入接收] 已收到输入: {text}")  # 调试日志
    #         except asyncio.CancelledError:
    #             break
    #         except Exception as e:
    #             print(f"文本输入错误: {str(e)}")
    #             self.running = False

    async def start_listening(self):
        """异步文本输入监听"""
        while self.running:
            try:
                # 使用aioconsole实现非阻塞输入
                # if multi_model.upload_file():
                #     text = multi_model.upload_file()[1]
                # else:
                text = await ainput("")
                await self.input_queue.put(text.strip())
                print(f"[文本输入接收] 已收到输入: {text}")  # 调试日志
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"文本输入错误: {str(e)}")
                self.running = False


class BedrockModelsWrapper:

    @staticmethod
    def define_body(text):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                body['prompt'] = f"""
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    Suppose you're a Chinese-proficient teacher and you're teaching a student who is learning Chinese and not know Chinese enough.
                    <|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    What can you help me with the following content. {text} Please output in Chinese.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
            else: 
                body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                # 读取所需的数据
                # Claude-3 的 message 信息
                mesg = ft_data['anthropic']['claude-3']['messages']
                # Claude-3的system信息
                system_mesg = ft_data['anthropic']['claude-3']['system']

                # 为了防止数据集中最后一个不是 assistant 的情况
                if mesg[-1]['role'] == 'assistant':
                    mesg.append({
                        "role": "user",
                        "content": text
                    })
                else:
                    # 数据集中最后一个是 user 的情况，直接将用户输入添加到最后一个 user 的 content 中
                    mesg[-1]['content'] += text

                # 数据导入 body 中
                body['messages'] = mesg
                if system_mesg:
                    body['system'] = system_mesg
            else:
                body['prompt'] = f'\n\nHuman: {text}\n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = text
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        else:
            raise Exception('Unknown model provider.')

        return body

    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]

        chunk_obj = ''
        text = ''

        # 添加知识库标记检测
        if "[官方知识库]" in text:
            print("\n[混合模式] 检测到知识库引用")
            return text.split("]", 1)[1].strip()  # 提取有效内容

        if model_provider == 'amazon':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
        elif model_provider == 'meta':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['generation']
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                # if chunk_obj['type'] == 'message_delta':
                #     print(f"\nStop reason: {chunk_obj['delta']['stop_reason']}")
                #     print(f"Stop sequence: {chunk_obj['delta']['stop_sequence']}")
                #     print(f"Output tokens: {chunk_obj['usage']['output_tokens']}")
                if chunk_obj['type'] == 'content_block_delta':
                    if chunk_obj['delta']['type'] == 'text_delta':
                        print(chunk_obj['delta']['text'], end="")
                        text = chunk_obj['delta']['text']
            else:
                #Claude2.x
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['completion']
        elif model_provider == 'cohere':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = ' '.join([c["text"] for c in chunk_obj['generations']])
        elif model_provider == 'mistral':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputs'][0]['text']
        else:
            raise NotImplementedError('Unknown model provider.')

        printer(f'[DEBUG] {chunk_obj}', 'debug')
        return text




def to_audio_generator(bedrock_stream):
    prefix = ''

    if bedrock_stream:
        for event in bedrock_stream:
            chunk = BedrockModelsWrapper.get_stream_chunk(event)
            if chunk:
                text = BedrockModelsWrapper.get_stream_text(chunk)

                if '.' in text:
                    a = text.split('.')[:-1]
                    to_polly = ''.join([prefix, '.'.join(a), '. '])
                    prefix = text.split('.')[-1]
                    print(to_polly, flush=True, end='')
                    yield to_polly
                else:
                    prefix = ''.join([prefix, text])

        if prefix != '':
            print(prefix, flush=True, end='')
            yield f'{prefix}.'

        print('\n')


class ZhipuKnowledgeManager:


    def __init__(self):
        self.api_key = self._get_zhipu_secret()  # 从AWS Secrets Manager获取
        self.knowledge_id = "1895725609330565120"  # 替换实际知识库ID
        self.min_similarity = 0.95  # 知识库匹配阈值

    def _get_zhipu_secret(self):
        """获取API密钥"""
        secret_name = os.getenv("ZHIPUAI_API_KEY")
        region_name = "us-east-1"

        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager",
            region_name=region_name
        )
        try:
            response = client.get_secret_value(SecretId=secret_name)
            return response["SecretString"]
        except Exception as e:
            raise RuntimeError(f"获取智谱密钥失败: {str(e)}")


    def query_knowledge(self, question):
        """调用智谱知识库API"""
        payload = {
            "model": "glm-4",
            "messages": [{
                "role": "user",
                "content": question,
                "retrieval_query": {  # 新增检索优化参数
                    "query_text": question,
                    "top_k": 3,
                    "score_threshold": self.min_similarity
                }
            }],
            "stream": True,

        "tools": [{
            "type": "retrieval",
            "retrieval": {
                "knowledge_id": self.knowledge_id,
                "prompt_template": (
                    "你是一个严谨的《红楼梦》研究助手，必须严格遵循以下要求：\n"
                    "1. 只能使用以下知识库内容回答问题，禁止使用外部知识\n"
                    "2. 如果知识库内容与问题无关，必须回答'根据现有资料无法回答该问题'\n"
                    "3. 回答必须包含具体出处（例：根据《红楼梦》第X回记载）\n"
                    "4. 保持口语化但不得编造信息\n\n"
                    "知识库内容：\n"
                    "'''\n{{knowledge}}\n'''\n\n"
                    "用户问题：{{question}}"
                ),
                "knowledge_config": {
                    "max_segment_length": 800,  # 控制知识片段长度
                    "score_type": "cosine"  # 优化相似度算法
                }
            }
        }]
        }

        try:
            response = requests.post(
                "https://open.bigmodel.cn/api/llm-application/open/knowledge/1895725609330565120'",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=8
            )
            response.raise_for_status()
            result = response.json()

            # 验证知识库调用有效性
            if result.get("usage", {}).get("retrieval_tokens", 0) > 0:
                print(f"[知识库命中] 使用token: {result['usage']['retrieval_tokens']}")
                return self._parse_response(result)
            else:
                print("[知识库未命中] 未找到相关段落")
                # 添加重试机制
                max_retries = 2
                for attempt in range(max_retries):
                    response = requests.post(...)
                    if 500 <= response.status_code < 600 and attempt < max_retries - 1:
                        print(f"服务端错误，正在重试 ({attempt + 1}/{max_retries})")
                        time.sleep(1.5 ** attempt)
                        continue
                    response.raise_for_status()
                    break

        except requests.exceptions.RequestException as e:

            print(f"[关键错误] 知识库请求失败: {str(e)}")

            return "service_unavailable"  # 特殊错误代码

        except json.JSONDecodeError:

            print("[格式错误] 无法解析响应JSON")

            return None

    def _parse_response(self, result):
        """增强版响应解析"""
        content = result["choices"][0]["message"]["content"]

        # 严格验证规则
        invalid_conditions = [
            "我不知道",
            "根据我的知识",
            "根据公开资料",
            not any(c in content for c in ["《红楼梦》", "第", "回"])]

        if any(condition in content for condition in invalid_conditions):
            print("[验证失败] 响应未正确引用知识库")
            return None

        # 检查引用格式
        if "根据《红楼梦》第" not in content and "原文提到" not in content:
            print("[格式警告] 未按规范标注出处")
            return content + "\n（以上信息来自《红楼梦》知识库）"

        return content


class BedrockWrapper:

    def __init__(self):
        self.speaking = False
        self.output_lock = asyncio.Lock()  # 添加输出锁

    async def safe_output(self, text, source_type):
        """线程安全的输出处理"""
        async with self.output_lock:
            # 语音输出
            if not self.speaking:
                self.invoke_bedrock(text)

            # 文字输出
            print(f"\n[AI响应/{source_type}] {text}\n> ", end='', flush=True)

    def is_speaking(self):
        return self.speaking

    def invoke_bedrock(self, text):
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        body = BedrockModelsWrapper.define_body(text)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )


            printer('[DEBUG] Capturing Bedrocks response/bedrock_stream', 'debug')
            bedrock_stream = response.get('body')
            printer(f"[DEBUG] Bedrock_stream: {bedrock_stream}", 'debug')

            audio_gen = to_audio_generator(bedrock_stream)
            printer('[DEBUG] Created bedrock stream to audio generator', 'debug')

            reader = Reader()
            for audio in audio_gen:
                reader.read(audio)

            reader.close()

        except Exception as e:
            print(e)
            time.sleep(3)
            self.speaking = False

        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')


class EnhancedBedrockWrapper(BedrockWrapper):
    def __init__(self):
        super().__init__()
        self.knowledge_manager = ZhipuKnowledgeManager()
        self.context_history = []  # 保存对话上下文

    def _build_prompt(self, text, knowledge):
        """构建增强提示词"""
        return f"""
        你是一个智能助手，并且对红楼梦非常有了解，请结合以下知识库内容和对话历史回答问题：

        [知识库内容]
        {knowledge}

        [对话历史]
        {self._format_history()}

        [当前问题]
        {text}
        
        [回答要求]
        1. 优先使用知识库内容
        2. 保持口语化中文
        3. 标明引用来源
        4. 如信息冲突，以知识库为准
        """

    def _format_history(self):
        """格式化历史对话"""
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in self.context_history[-5:]])  # 保留最近5轮对话

    def invoke_bedrock(self, text):
        """混合知识库调用流程"""
        # 步骤1：优先查询知识库
        knowledge_response = self.knowledge_manager.query_knowledge(text)

        # 步骤2：根据结果选择处理路径
        if knowledge_response:
            # 知识库模式：直接使用智谱响应
            print(f"\n[知识库模式] 响应内容：{knowledge_response[:200]}...")
            self._voice_output(knowledge_response)
            return knowledge_response
        else:
            # 基础模式：调用Bedrock生成
            print("\n[基础模式] 调用Bedrock生成...")
            return super().invoke_bedrock(text)
        # 保存对话上下文
        self.context_history.append((text, response))
        return response[-1]

    async def invoke_model(self, input_payload):
        """支持多模态的模型调用"""
        body = self._build_multimodal_body(input_payload)

        response = bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            accept='application/json',
            contentType='application/json'
        )

        return self._parse_model_response(response)

    def _build_multimodal_body(self, input_data):
        """构建符合Claude 3多模态输入的请求体"""
        messages = []

        if input_data['type'] == 'text':
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": input_data['content']}]
            })
        elif input_data['type'] == 'image':
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": input_data['content']
                    }},
                    {"type": "text", "text": input_data['analysis']['question']}
                ]
            })

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": messages,
            "temperature": 0.5
        }


class Reader:

    def __init__(self):
        self.polly = boto3.client('polly', region_name=config['region'])
        self.audio = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        self.chunk = 1024

    def read(self, data):
        response = self.polly.synthesize_speech(
            Text=data,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )

        stream = response['AudioStream']

        while True:
            # Check if user signaled to shut down Bedrock speech
            # UserInputManager.start_shutdown_executor() will raise Exception. If not ideas but is functional.
            if UserInputManager.is_executor_set() and UserInputManager.is_shutdown_scheduled():
                UserInputManager.start_shutdown_executor()

            data = stream.read(self.chunk)
            self.audio.write(data)
            if not data:
                break

    def close(self):
        time.sleep(1)
        self.audio.stop_stream()
        self.audio.close()


class EnhancedReader:
    def __init__(self):
        self._active = False

    def read(self, data):
        """支持中断的读取方法"""
        self._active = True
        try:
            response = self.polly.synthesize_speech(...)
            stream = response['AudioStream']

            while self._active and not UserInputManager._stop_event.is_set():
                data = stream.read(1024)
                if not data:
                    break
                self.audio.write(data)

        finally:
            self.cleanup()

    def cleanup(self):
        """资源清理"""
        if self._active:
            self.audio.stop_stream()
            self.audio.close()
            self._active = False
            UserInputManager._stop_event.clear()  # 重置状态

    def stop(self):
        """外部停止方法"""
        self._active = False

class AudioController:
    def stop_audio(self):
        """统一停止入口"""
        UserInputManager._stop_event.set()
        if self.current_reader:
            self.current_reader.stop()

# 初始化时绑定实例
audio_controller = AudioController()
UserInputManager.bind_audio_controller(audio_controller)

last_time = 3
class EventHandler(TranscriptResultStreamHandler):
    text: list[str] = []
    sample_count = 0
    max_sample_counter = 4

    def __init__(self, transcript_result_stream: TranscriptResultStream, bedrock_wrapper):
        super().__init__(transcript_result_stream)
        self.bedrock_wrapper = bedrock_wrapper
        self.input_text = ""  # 初始化类属性

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if not self.bedrock_wrapper.is_speaking():
            if results:
                for result in results:
                    EventHandler.sample_count = 0
                    if not result.is_partial:
                        for alt in result.alternatives:
                            transcript = alt.transcript.strip()
                            if transcript:  # 添加空值检查
                                print(f"[语音识别] 收到输入: {transcript}", flush=True)
                                self.input_text += transcript + " "  # 使用类属性存储


            else:
                EventHandler.sample_count += 1
                if EventHandler.sample_count == EventHandler.max_sample_counter:

                    # if len(EventHandler.text) == 0:
                    #     last_speech = config['last_speech']
                    #     print(last_speech, flush=True)
                        #aws_polly_tts(last_speech)
                        #os._exit(0)  # exit from a child process
                    #else:
                    if self.input_text.strip():
                        printer(f'\n[INFO] 最终输入文本: {self.input_text}', 'info')
                        executor = ThreadPoolExecutor(max_workers=1)
                        UserInputManager.set_executor(executor)
                        loop.run_in_executor(
                            executor,
                            self.bedrock_wrapper.invoke_bedrock,
                            self.input_text.strip()  # 使用清理后的文本
                        )
                        time.sleep(last_time)
                    else:
                        time.sleep(last_time)
                        print("[警告] 未检测到有效语音输入", flush=True)

                        # 重置状态
                    self.input_text = ""
                    EventHandler.sample_count = 0




class MicStream:

    async def mic_stream(self):
        print("[语音输入] 麦克风准备就绪...")  # 调试日志
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        try:
            stream = sounddevice.RawInputStream(
                channels=1,
                samplerate=16000,
                callback=callback,
                blocksize=2048 * 2,
                dtype="int16"
            )
            with stream:
                while True:
                    indata, status = await input_queue.get()
                    yield indata, status
        except Exception as e:
            print(f"麦克风错误: {str(e)}")
            raise

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        await stream.input_stream.end_stream()

    async def basic_transcribe(self):
        loop.run_in_executor(ThreadPoolExecutor(max_workers=1), UserInputManager.start_user_input_loop)

        stream = await transcribe_streaming.start_stream_transcription(
            language_code="zh-CN",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        handler = EventHandler(stream.output_stream, BedrockWrapper())
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())


class DualInputHandler:
    def __init__(self, bedrock_wrapper):
        self.bedrock = bedrock_wrapper
        self.text_manager = TextInputManager()
        self.mic_manager = MicStream()

    async def handle_text_input(self):
        """处理文本输入通道"""
        while True:
            try:
                text = await self.text_manager.input_queue.get()
                if text.lower() == 'exit':
                    print("正在关闭程序...")
                    os._exid(0)

                print(f"\n[处理文本输入] 正在处理: {text}")
                await self.process_input(text, "文本")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"文本处理错误: {str(e)}")

    async def handle_voice_input(self):
        """处理语音输入通道"""
        try:
            stream = await transcribe_streaming.start_stream_transcription(
                language_code="zh-CN",
                media_sample_rate_hz=16000,
                media_encoding="pcm",
            )
            handler = EventHandler(stream.output_stream, self.bedrock)

            await asyncio.gather(
                self.mic_manager.write_chunks(stream),
                handler.handle_events()
            )
        except Exception as e:
            print(f"语音处理错误: {str(e)}")

    async def process_input(self, text, input_type):
        """统一处理输入内容"""
        try:
            # 显示输入反馈
            print(f"\n[{input_type}输入] 正在生成响应...")

            # 调用Bedrock生成响应
            executor = ThreadPoolExecutor(max_workers=1)
            await loop.run_in_executor(
                executor,
                self.bedrock.invoke_bedrock,
                text
            )

            print(f"\n[{input_type}输入] 处理完成")

        except Exception as e:
            print(f"处理异常: {str(e)}")



async def main():
    bedrock_wrapper = BedrockWrapper()
    handler = DualInputHandler(bedrock_wrapper)

    # 启动双通道监听
    input_tasks = [
        asyncio.create_task(handler.text_manager.start_listening()),
        asyncio.create_task(handler.handle_text_input()),
        asyncio.create_task(handler.handle_voice_input())
    ]

    # 监控任务状态
    while True:
        await asyncio.sleep(1)
        for task in input_tasks:
            if task.done():
                print(f"输入通道异常终止: {task.exception()}")
                os._exit(1)



info_text = f'''
*************************************************************
[INFO] Supported FM models: {get_model_ids()}.
[INFO] Change FM model by setting <MODEL_ID> environment variable. Example: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock model: {config['bedrock']['api_request']['modelId']}
[INFO] Polly config: engine {config['polly']['Engine']}, voice {config['polly']['VoiceId']}
[INFO] Log level: {config['log_level']}

[INFO] Hit ENTER to interrupt Amazon Bedrock. After you can continue speaking!
[INFO] Go ahead with the voice chat with Amazon Bedrock!
*************************************************************
'''

print(info_text)

if __name__ == "__main__":
    print("""
    *************************************************************
    [系统已升级]
    改进功能：
    1. 可靠的双通道输入监听
    2. 实时输入状态反馈
    3. 异常自动恢复机制

    操作指南：
    - 语音输入：直接说话（需确保麦克风权限）
    - 文字输入：输入文字后按回车
    - 输入 exit 回车可退出
    *************************************************************
    """)

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n安全关闭中...")
        sys.exit(0)
