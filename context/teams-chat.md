
3 new notifications
Has context menu
Chat




Unread
Channels
Chats

Unread messageLast messageGroup chatMeeting chatChatPersonal at mentionEveryone at mentionImportantUrgentDraftDraftMutedMeeting in progressMeet now in progressYou can't send messages because you are not a member of the chat.You cannot send messages to this botPrivateSharedHas context menuChannel at mentionTeam at mentionPersonal at mentionUnreadUnreadMeeting in progressUnreadChannelTeamHas pinned messagesSee moreCommunityTemporarily shownHas context menuBadged chatBadged chats
Has context menu

Teams and channels

Anthony
Blu
Nikhil


Chat

Shared

Recap

Has context menu

Meet now


4




Message List

Nikhil Gopal
The key points we'd hammer home in the demo: • Azure works together as a cohesive stack for the end-to-end LLM lifecycle • Don't always default to a managed endpoint. My 14b fine tune beat GPT 5.4 on…
5/26

It's about to launch so dunno how cooked it... by Anthony Nevico
Anthony Nevico
6/2 2:44 PM

It's about to launch so dunno how cooked it is

But could be cool by Anthony Nevico
Anthony Nevico
6/2 2:44 PM

But could be cool

Its cooked. Cursor's composer launched on i by Nikhil Gopal
Nikhil Gopal
6/2 2:45 PM

Its cooked. Cursor's composer launched on it

I subscribe to an AI newsletter and I've he... by Nikhil Gopal
Nikhil Gopal
6/2 2:45 PM

I subscribe to an AI newsletter and I've heard of these guys

DeepSeek v4 pro is currently 5$ per 1M outp... by Nikhil Gopal
Nikhil Gopal
6/2 2:45 PM

DeepSeek v4 pro is currently 5$ per 1M output tokens. They're offering 3.48. That's quite significant

Blu Gotlieb added Copilot here.
This message has been deleted. by Copilot

This message has been deleted.
<0></0> removed <1></1> from the chat.
Nikhil Gopal  - Do you guys want to shoot f... by Anthony Nevico
Anthony Nevico
6/9 3:26 PM

Nikhil Gopal - Do you guys want to shoot for a session on 6/26?  Sean and folks had to bail so looking to fill that slot, also have Pete who wants to demo an RLS thing

Yes let’s but the agenda I DMd you was targ... by Nikhil Gopal
Nikhil Gopal
6/9 3:48 PM

Yes let’s but the agenda I DMd you was targeting an hour, shall I cut it down then?
nah I'll give you the hour by Anthony Nevico
Anthony Nevico
6/9 4:39 PM

nah I'll give you the hour

Ok cool by Nikhil Gopal
Nikhil Gopal
Tuesday 5:57 PM

Ok cool

But still plan for 26th? by Nikhil Gopal
Nikhil Gopal
Tuesday 5:57 PM

But still plan for 26th?

yes please by Anthony Nevico
Anthony Nevico
Wednesday 9:42 AM

yes please

Time Segment Purpose 0–5 min Goal framing M... by Nikhil Gopal
Nikhil Gopal
Wednesday 10:48 AM

Time	Segment	Purpose
0–5 min	Goal framing	Model choice is a lifecycle decision: quality, latency, cost, safety, drift, and retraining readiness.
5–10 min	Start with a frontier model	Begin with Claude Opus / GPT-5.5-class models to establish a high-quality baseline and collect traces.
10–18 min	Use case: WebIQ research agent	WebIQ creates a strong drift scenario because live web/news/image/video grounding changes input distribution over time.
18–30 min	Golden dataset creation	Capture prompts, WebIQ retrieval context, completions, citations, human ratings, failure labels, latency, and token cost.
30–42 min	LLMOps evaluation pipeline	Run continuous and batch evaluations in Azure AI Foundry: groundedness, task adherence, fluency, safety, citation quality, latency, and cost.
42–52 min	Automatic retraining / model customization	Use Foundry model customization: SFT/distillation, DPO preference optimization, or RFT with graders depending on available feedback and task complexity.
52–58 min	Architecture walkthrough	Show how Foundry hosted agents, Fabric storage, model checkpoints, eval results, and drift monitoring work together.
58–60 min	Decisions and next steps	Define customer next steps: trace capture, golden set, eval gates, retraining trigger, deployment promotion criteria.
Architecture: AI Provider: Microsoft Foundr... by Nikhil Gopal
Nikhil Gopal
Wednesday 10:48 AM

Architecture:

AI Provider: Microsoft Foundry
Agent Runtime: Foundry Hosted Agents
Evals & Tracing: Foundry Tracing
Model Checkpoint Storage: Azure Blob Storage
Evaluation Run Results & Prompt Completions: Azure SQL DB
Persistent Agent Memory: Foundry IQ/Cosmos DB
Eventhouse (failures, latency, drift signals): Eventhouse

Foundry model lifecycle: customers start with a frontier model, collect production-quality traces, build a golden dataset in Fabric, and evaluate candidate models in Azure AI Foundry. When quality/cost/latency targets justify optimization, the pipeline uses Foundry fine-tuning capabilities such as SFT distillation, DPO, or RFT with graders to train a smaller or specialized model.

LLMOps loop: continuous evaluation watches hosted-agent responses. If quality drops, drift increases, WebIQ query patterns shift, or cost/latency exceeds thresholds, the pipeline automatically creates a refreshed dataset from Fabric, launches a new customization run, evaluates the candidate against the golden set plus recent drift samples, stores results/checkpoints back in Fabric, and only promotes the new model if it passes predefined quality, safety, latency, and cost gates.
Here’s a draft. Some content also needs to ... by Nikhil Gopal
Nikhil Gopal
Wednesday 10:49 AM

Here’s a draft. Some content also needs to be cut to allow for more q&a and discussion
We are by no means married to this agenda a... by Nikhil Gopal
Nikhil Gopal
Wednesday 10:49 AM

We are by no means married to this agenda and I welcome both of your suggestions. Blu I am also less fluent in fabric so pls correct if I made mistakes
Looking fwd to discussing on Fri by Nikhil Gopal
Nikhil Gopal
Wednesday 10:50 AM

Looking fwd to discussing on Fri
I applied for preview access to both WebIQ ... by Nikhil Gopal
Nikhil Gopal
Wednesday 10:53 AM

I applied for preview access to both WebIQ and the RL environments announced at build. It would be cool but if they don’t get approved we can fallback to the normal bing grounding and some other method of training
For the purposes of this demo I think the e... by Nikhil Gopal
Nikhil Gopal
Wednesday 10:53 AM

For the purposes of this demo I think the exact implementation details are less crucial than the concepts
By wed:   Jose: Define mock scenario: what ... by Nikhil Gopal
Nikhil Gopal
Friday 1:35 PM

By wed:

 

Jose:

Define mock scenario: what are they using the deployed LLM for, who are the users
Identify golden dataset and eval metric to benchmark frontier model vs smaller OSS model against
Identify new data to fine tune on (can be outside dataset with useful info to train on + frontier model's completions from prod traces)
Can also distill the frontier model's responses instead of fine tuning (Introducing Model Distillation in Azure OpenAI Service | Microsoft Community Hub)
Create fine-tune/distillation loop
If there isn't an easy abstraction in foundry, provision a GPU instance and code a LoRa or other PEFT method for a training loop
Blu:

Investigate best database option > provision & setup
Investigate Event Hub
Simple dashboarding solution to visualize model performance & drift over time (maybe PBI)
By Fri:

Collaborate to make the loop run automatically & continuously re-train or distill
Wire the foundry traces, prompts & completions & eval results into the database
Visualize results on dashboards
Second week:

Get anthony feedback
Implement results
Create & practice presentation
Url Preview for Introducing Model Distillation in Azure OpenAI Service | Microsoft Community Hub
Introducing Model Distillation in Azure OpenAI Service | Microsoft Community Hub
  We're excited to introduce our upcoming release of Model Distillation feature in Azure OpenAI Service. This feature provides developers with a...

techcommunity.microsoft.com

🫡
1 Saluting face reaction.

Some notes above, thanks sm for your help g... by Nikhil Gopal
Nikhil Gopal
Friday 1:36 PM

Some notes above, thanks sm for your help guys. Let's knock this out of the park

Jose I wasn't aware foundry natively suppor... by Nikhil Gopal
Nikhil Gopal
Friday 1:36 PM

Jose I wasn't aware foundry natively supported distillation too, I haven't had a chance to play with it yet. But it could even be better than fine tuning. Take a look and I leave it to your discretion which to choose


👍
1 Like reaction.

We can also address it in the presentation ... by Nikhil Gopal
Nikhil Gopal
Friday 1:36 PM

We can also address it in the presentation and explain the difference, and you can argue why you choose what you did, or both if relevant

has context menu
Status of Nikhil Gopal: On site with customer - responses delayed


