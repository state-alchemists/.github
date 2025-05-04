# Metrics + Principle vs Goal


Someday, you come to your stakeholders, bringing your KPI and metrics. Your KPI was carefully crafted based on some well-known principles out there. You lead an engineering team, all your services are well monitored, 100% code coverage, 99.99% SLA. No room for error. Yet, despite of good numbers you present, your stakeholder doesn't seem to be happy.

The stakeholders are questioning you about why the IT team needs hundreds of dollars to keep the server running. You hold your breath for a while. It must be very obvious in your presentation already. It is all needed for the good metrics you have just presented earlier. But you patiently educate the stakeholders. It is all important.

The stakeholder then complains about why the revenue doesn't go up. You start losing your temper. It is not your team's fault. The stakeholders should ask the marketing and sales team, not you. And you start to wonder how stakeholders could be so stupid.

# Metrics vs Goal

Your stakeholders doesn't care about all the metrics. He doesn't care whether your system is 100% available or not, the most important thing for them is the cashflow, that the business is running.

Only you, the middle management, care about KPI. You need those numbers to defend yourself, to show that you have done your job very well, to justify whether you need more people or you need to fire the existing ones.

In my last meeting, I asked my boss, "What is your expectation? What is your goal?" While my peers are thinking about KPI and how measurable things should be, I don't care that much. Yes, I know, I should care more, and good metrics help us to measure our performance, but I can't help but not care that much. I care about the goal more. The goal can usually be summed up in a couple of sentences or a single paragraph.

My KPI can't be bad as long as I help my team achieve the goal. If it's bad, then the KPI is wrong. As long as I'm useful, my boss won't let me go. On the other hand, if my KPI was excellent but I didn't really contribute anything, then my boss will seek a way to manage me out.

That's how I see the current trend of "silent layoffs." Some of my LinkedIn connections, truly masterful software engineers with moonshot KPI, are getting fired because of the recent economic condition. It is not really their fault, right?

Yes, no one is at fault here. The reality is much harsher. They are not needed. Despite their good KPI, their contribution to achieving the company goal is probably not that much.

# Principle vs Goal

The software engineering landscape is full of buzzwords, jargon, and principles. SOLID principle, DRY, whatever. Moreoften, our KPI was based on these principles. We have checklists, we have 12 factor app or something. Our service should be as stateless as possible, so that even if I only need a single container, and I need to save some state, I need to use Redis/DB, do an entire network call that most likely slower than accessing a local JSON file.

Now, let me show you a not-so-pythonic code.

```
async def retrieve(query: str) -> str:
    from chromadb import PersistentClient
    from chromadb.config import Settings
    from openai import OpenAI
    ... 
```

Most Python developer will scrutinize the code, saying that `import` should be located at the beginning of the file. Putting import inside the functions will make the dependencies less obvious and probably will make the code even longer in case of you have multiple functions importing the same thing.

The statement wasn't wrong at all, and it is especially true if your application is long-running application. We can afford a bit overhead at the beginning as long as the dependencies are loaded into the memory and ready to be used at any point.

But let me show you [this issue on Zrb](https://github.com/state-alchemists/zrb/issues/204). Zrb is a CLI tool. It means that it will be started several times, and eager loading things that less likely used can severe performance greatly.

```
-> % python -m benchmark_imports zrb
4.4327 root       zrb
2.7971 project    zrb.builtin
0.9076 project    zrb.llm_config
0.6143 dependency pydantic_ai.models.openai
0.6083 transitive openai                                   from pydantic_ai.models.openai
0.4013 project    zrb.builtin.git
0.3888 project    zrb.builtin.todo
0.3456 transitive openai._client                           from openai
0.3424 transitive openai.resources                         from openai._client
0.3422 project    zrb.task.llm_task
0.3382 project    zrb.builtin.setup.asdf.asdf
0.3230 project    zrb.builtin.setup.latex.ubuntu
0.3195 dependency pydantic_ai.mcp
0.3107 transitive mcp                                      from pydantic_ai.mcp
0.2897 dependency pydantic_ai
0.2769 project    zrb.builtin.setup.ubuntu
0.2716 transitive pydantic_ai.agent                        from pydantic_ai
0.2592 transitive openai.types                             from openai
0.2465 project    zrb.builtin.setup.zsh.zsh
0.2067 transitive mcp.client.session                       from mcp
0.2064 project    zrb.callback.callback
0.2050 project    zrb.session.any_session
0.2016 project    zrb.session_state_log.session_state_log
0.1926 transitive mcp.types                                from mcp.client.session
0.1780 transitive openai.resources.beta                    from openai.resources
```

A CLI tool, spending 4 seconds just to import things is not acceptable. So, I implement lazy loading, putting imports inside functions and methods, make sure they will only be loaded when necessary.

```
0.6412 root       zrb
0.4397 project    zrb.builtin
0.1232 project    zrb.callback.callback
0.1221 project    zrb.session.any_session
0.1196 project    zrb.session_state_log.session_state_log
0.0616 project    zrb.builtin.todo
0.0543 dependency pydantic
0.0493 project    zrb.builtin.setup.asdf.asdf
0.0482 transitive pydantic._migration                      from pydantic
0.0480 transitive pydantic.version                         from pydantic._migration
0.0479 transitive pydantic_core                            from pydantic.version
0.0441 transitive pydantic_core.core_schema                from pydantic_core
0.0439 project    zrb.builtin.setup.latex.ubuntu
0.0439 project    zrb.builtin.git
0.0380 project    zrb.runner.cli
0.0349 project    zrb.builtin.setup.zsh.zsh
0.0330 project    zrb.builtin.setup.ubuntu
0.0298 dependency pydantic.main
0.0291 project    zrb.builtin.project.add.fastapp.fastapp_task
0.0272 project    zrb.builtin.setup.tmux.tmux
0.0272 project    zrb.builtin.git_subtree
0.0253 transitive asyncio                                  from pydantic_core.core_schema
0.0222 transitive asyncio.base_events                      from asyncio
0.0221 project    zrb.builtin.base64
0.0188 transitive pydantic._internal._model_construction   from pydantic.main
```

The result? 0.64 second. Much better.

The goal is to make a usable tool, even if it has to defy some principles.

# How We Should See Things

**Goal is your ultimate destination**. It is something you should worship with all your dedication, hardwork, life and soul.

**OKR and KPI** should be measurable, KPI **shows you how much you have done or how close you are to your goal**. If it doesn't work that way, it is wrong. KPI always started good, but along the time it become game. People chasing for numbers, for how many documents they produce in every quarter. When your OKR/KPI doesn't reflect your goal, you need to reevaluate it.

Principle is a **heuristic** to achieve your goal. In most cases, when you follow the principles, you can't go wrong. But there are edge cases. When you use certain principles to build your solution, make sure it plays well with the goal, not just because Uncle Bob tells you do so. Uncle Bob won't help you if things goes wrong. He is not even there when you are fired by the stakeholders.
