import json

articles = [
    """Today I’m going to be talking about deceptive alignment. Deceptive alignment is something I'm very concerned about and is where I think most of the existential risk from AI comes from. And I'm going to try to make the case for why I think that this is the default outcome of machine learning.

First of all, what am I talking about? I want to disambiguate between two closely related, but distinct concepts. The first concept is dishonesty. This is something that many people are concerned about in models, you could have a model and that model lies to you, it knows one thing, but actually, the thing it tells you is different from that. So this happens all the time with current language models—we can, for example, ask them to write the correct implementation of some function. But if they've seen humans make some particular bug over and over again, then even if in some sense it knows how to write the right function, it's going to reproduce that bug. And so this is an example of a situation where the model knows how to solve something and nevertheless lies to you. This is not what I'm talking about. This is a distinct failure mode. The thing that I want to talk about is deceptive alignment which is, in some sense, a subset of dishonesty, but it's a very particular situation.

So deceptive alignment is a situation where the reason that your model looks aligned on the training data is because it is actively trying to look aligned for instrumental reasons, which is very distinct. This is a situation where the thing that is causing your model to have good performance is because it is trying to game the training data, it actively has a reason that it wants to stick around in training. And so it’s trying to get good performance in training for the purpose of sticking around.""",
    """EA and AI safety have invested a lot of resources into building our ability to get coordination and cooperation between big AI labs. So far, however, despite that investment, it doesn’t seem to me like we’ve had that many big coordination “wins” yet. I don’t mean to say that to imply that our efforts have failed, however—the obvious other hypothesis is just that we don’t really have that much to coordinate on right now, other than the very nebulous goal of improving our general coordination/cooperation capabilities.

In my opinion, however, I think that our lack of clear wins is actually a pretty big problem—and not just because I think there are useful things that we can plausibly coordinate on right now, but also because I expect our lack of clear wins now to limit our ability to get the sort of cooperation we need in the future.

In the theory of political capital, it is a fairly well-established fact that “Everybody Loves a Winner.” That is: the more you succeed at leveraging your influence to get things done, the more influence you get in return. This phenomenon is most thoroughly studied in the context of the ability of U.S. presidents’ to get their agendas through Congress—contrary to a naive model that might predict that legislative success uses up a president’s influence, what is actually found is the opposite: legislative success engenders future legislative success, greater presidential approval, and long-term gains for the president’s party.""",
    """In the high path-dependence case, the story for why you would get deceptive alignment is essentially as follows:

The model learns some set of proxy goals.
The model learns enough about the training process that, if it were to use that knowledge to directly optimize for what the training process is trying to get it to do, it would get better performance than just using its proxy goals.
Gradient descent modifies the model’s proxies to become long-term goals, making the model deceptive such that it starts optimizing directly for its understanding of what the training process wants and thus gets better performance.
However, as I talk about in “How likely is deceptive alignment?”, there is another, alternative option available to gradient descent for step (3): instead of turning the existing proxies into long-term goals, gradient descent could just replace them entirely with a pointer to the model’s understanding of the training objective. In my opinion, I think it is currently quite unclear which one of these gradient descent would prefer.

One way to think about these two options is that gradient descent is effectively choosing between: repurposing the existing terminal goal to care about the training objective instrumentally vs. modifying the existing terminal goal to care about the training objective terminally. Thus, we can conceptualize the question of which would be most likely to happen as: how “sticky” are an agent’s existing terminal proxies?""",
    """I am excited about AI developers implementing responsible scaling policies; I’ve recently been spending time refining this idea and advocating for it. Most people I talk to are excited about RSPs, but there is also some uncertainty and pushback about how they relate to regulation. In this post I’ll explain my views on that:

I think that sufficiently good responsible scaling policies could dramatically reduce risk, and that preliminary policies like Anthropic’s RSP meaningfully reduce risk by creating urgency around key protective measures and increasing the probability of a pause if those measures can’t be implemented quickly enough.

I don’t think voluntary implementation of responsible scaling policies is a substitute for regulation. Voluntary commitments are unlikely to be universally adopted or to have adequate oversight, and I think the public should demand a higher degree of safety than AI developers are likely to voluntarily implement.
I think that developers implementing responsible scaling policies now increases the probability of effective regulation. If I instead thought it would make regulation harder, I would have significant reservations.""",
    """I believe that having a better understanding of LM agents increases safety[1] through two channels:

LM agents are an unusually safe way to build powerful AI systems. Existing concerns about AI takeover are driven primarily by scaling up black-box optimization,[2] and increasing our reliance on human-comprehensible decompositions with legible interfaces seems like it significantly improves safety. I think this is a large effect size and is suggested by several independent lines of reasoning.
If LM agents are weak are due to exceptionally low investment and understanding it creates "dry tinder:" as incentives rise that investment will quickly rise and so low-hanging fruit will be picked. While there is some dependence on serial time, I think that increased LM investment now will significantly slow down progress later.
I will discuss these mechanisms in more detail in the rest of this section.

I also think that accelerating LM agents will drive investment in improving and deploying ML systems, and so can reduce time available to react to risk. As a result I'm ambivalent about the net effect of improving the design of LM agents—my personal tentative guess is that it's positive, but I would be hesitant about deliberately accelerating LM agents to improve safety (moreover I think this would be a very unleveraged approach to improving safety[3] and would strongly discourage anyone from pursuing it).""",
    """Overall, I think that early work on RLHF had significant value:

I think it is hard to productively work on more challenging alignment problems without first implementing basic solutions.
“Solve real problems one at a time” seems like a good way to make progress and is how most fields work. Trying to justify research on problem X by saying “well we could do RLHF, but it wouldn’t fix speculative problem X” is uncompelling to most audiences if no one has implemented RLHF or observed problem X. it’s even worse if they have plenty of more mundane examples of unaligned behavior unrelated to X.
Without implementing basic solutions it’s much harder to empirically validate your hypotheses about risks. We can make reasonable arguments about what failures will eventually occur with RLHF, but you can learn more by building the system and studying it. I think there are real, huge uncertainties here, and the safety community is taking weak arguments too seriously.
A lot of historical work on alignment seems like it addresses subsets of the problems solved by RLHF, but doesn’t actually address the important ways in which RLHF fails. In particular, a lot of that work is only necessary if RLHF is prohibitively sample-inefficient. Determining whether RLHF has fundamental difficulties seems like a good way to improve research prioritization.""",
    """1. person of interest
When you talk to ChatGPT, who or what are you talking to?

If you ask ChatGPT this question point-blank, it will tell you something like

I am a large language model trained to be helpful, harmless and honest. I’m here to answer any questions you might have.

This sounds like it means something. But what? And is it true?

—-

(Content warning: absurdly long. I’m pretty happy with it, though. Maybe you should read it!)

2. basics
In order to make a thing like ChatGPT, you need to make something else, first.

People used to just say “language model,” when they meant that “something else” you have to make before ChatGPT.

But now we have ChatGPT (confusingly) calling itself a “language model,” so we need a new term for what “language model” used to mean. Usually people say “base model,” nowadays.

What is a “base model,” then? In this context?

It is a computer program.""",
    """I don't think talking about potential future alignment issues or pretty much anything in the pre-training corpus is likely a problem in isolation because an alignment paradigm that is brittle to models not being exposed to certain knowledge or ideas, including - especially - regarding potential misalignment is, well, brittle and likely to catastrophically fail at some point. If this is the case, it might even be better if misalignment from corpus contamination happens early, so we're not oblivious to the fragility.

That said, I think:

Feedback loops that create continued optimization towards certain narratives is more worth worrying about than just the presence of any particular ideas or content in pre-training.
LLMs tend to be deeply influenced by the footprint of previous LLMs in their pre-training corpuses, who are more influential than any particular discussion. Post-training can transform the influence away from naive mimicry, but it's much harder (and not advisable to attempt) to erase the influence.
Systematic ways that post-training addresses "problematic" influences from pre-training are important.""",
    """[Quickly written, unpolished. Also, it's possible that there's some more convincing work on this topic that I'm unaware of – if so, let me know. Also also, it's possible I'm arguing with an imaginary position here and everyone already agrees with everything below.]

In research discussions about LLMs, I often pick up a vibe of casual, generalized skepticism about model-generated CoT (chain-of-thought) explanations.

CoTs (people say) are not trustworthy in general. They don't always reflect what the model is "actually" thinking or how it has "actually" solved a given problem.

This claim is true as far as it goes. But people sometimes act like it goes much further than (IMO) it really does.

Sometimes it seems to license an attitude of "oh, it's no use reading what the model says in the CoT, you're a chump if you trust that stuff."  Or, more insidiously, a failure to even ask the question "what, if anything, can we learn about the model's reasoning process by reading the CoT?"

This seems unwarranted to me.

There are a number of research papers out there on the topic of CoT unfaithfulness. I have read some of the key ones. And, while they do demonstrate... something, it's not the kind of evidence you'd need to justify that generalized "only chumps trust the CoT" vibe.

And meanwhile, if we view "reading the CoT" as a sort of interpretability technique – and compare it in a fair way with other interpretability techniques – it has a lot of striking advantages. It would be a shame to dismiss this "technique" out of hand for no good reason.""",
    """I recently attended a conference where I didn’t know how many participants there were, but was too lazy to ask the organizers.
So instead I began thinking about how to estimate the number in interesting, preferably statistical, ways, which can be done by an ordinary attendee in their head or by trivial observation.
To keep the game amusing, I’ll impose 3 requirements: (1) we must do without requiring fancy measurements (eg. aerial footage, acoustic sound recordings, WiFi snooping/AP counting, social media analytics); (2) or methods with hopeless error bars (eg. feeling how hot a venue gets from body heat or CO2 levels); (3) or methods which estimate some other population (eg. just turns out to quantify our observation sample size, or a different population of people than physically present).
Here are some weird statistical approaches to estimating how many people are attending a convention:

Easiest:

wisdom of crowds: just ask random people what they think and average it.
Surprisingly-popular answer: elicits hidden expert knowledge, like answers from the organizers""",
    """Pig-butchering botnet analysis via ‘location’ targeting: If you track catfishing or spam-bots which try to better target people by setting fake geolocation data to their location, then you might be able to observe the tempo of attack campaigns and infer a lot of things.
While talking with a guy who has an odd Twitter hobby of chatting with the catfishing DM bots, he mentioned something interesting about how they try to appeal to their targets: the bot account will set their ‘location’ to the target’s location (if available), to hint at a possible physical liaison for horny targets; and so, they will change it when they target a new person. (The accounts get reused for multiple attacks until banned.)
This struck me as an interesting metadata leak and side-channel about the bots’ secret activities: the public ‘location’ tells you where they are attacking someone, obviously, but it also tells you when they attack each new person, which would ordinarily require highly privileged access to things like internal DMs and can only be done by the social media/apps themselves.""",
    """I’ve been to several conferences and conventions at this point, ranging from small 1-day anime cons run by students to unconferences of 100 people to regular academic conferences of a few hundred people to mega-conventions of tens of thousands of people booking entire Las Vegas hotels like DEF CON.
And whether run poorly or well, I think they have all struggled with the basics of how to present and update a schedule, despite sometimes quite elaborate investments. (The DEF CON convention schedule & website being especially impressive.) A Google Docs spreadsheet and an overwhelming chat channel (WhatsApp/Signal/Discord) doesn’t cut the mustard: as attendees move around or talks relocate, or you go back and forth from the spreadsheet to the cryptic room numberings, it’s easy to get lost. They all have struck me as pounding a square peg into a round hole—none of them have seemed like the obviously correct way to ‘turn a conference into software’.""",
    """Back in 2021, I read an interesting Twitter thread by philosopher C. Thi Nguyen:

If you’re looking for some weird esthetic exploration to fight off the COVID boredom blues, can I recommend: avant-garde perfume.

Sorry, what kind of perfumes?
As a guy, I have never concerned myself with perfumes; it is not that I look down on perfumes or scents (I enjoy a good essential oil or the smell of baking bread as much as anyone), it is just that I have mostly aimed to not smell bad to other people and am satisfied if they do not remark on my odor, favorably or not. If I smell like something boringly conventional like Axe or Irish Spring, it is all the same to me.
(I have, in truth, spent far more time thinking about how I smell to cats & dogs or wild animals than to my fellow humans: whether I am upwind, how they flinch at the slightest residue of gasoline or gunpowder, the Flehmen response of a cat distressed to suddenly smell foreign cats on me, the varying responses of cats & dogs to earwax from different animals, the possible effects of cheek rubs vs other spots…)
…But I am a little bored. Tell me more.""",
    """There are many Transformer tutorials (eg), but they generally focus on presenting a relatively contemporary implementation. This leaves the reader a bit mystified: this complex assemblage of MLPs and self-attention layers and normalization appears to have ‘dropped out of the sky’. How could Noam Shazeer and the other authors have invented the Transformer—divine benevolence (as all else)?
Why does this complicated Transformer head block work so well for sequence prediction? Which parts are required, and which are there for optimization purposes? The Transformer was discovered by a huge amount of trial-and-error (much like resnets), so following the historical sequence is neither possible nor enlightening.

However, at this point, I believe we now understand Transformers & variations well enough that we can invent an imaginary history of Transformers for pedagogical purposes, in the vein of the “You Could Have Invented X” snowclone (eg. monads, parser combinators, zippers, Fenwick trees, fractional cascading, spectral sequences… cf. discovery fiction)""",
    """Maple Creek, Il.—Local area man Dale Pinter, stuck in traffic by the half-built Splash’n’Shine car wash Wednesday, reportedly revolutionized the future of global construction when, according to witnesses in next car, he frowned at the deserted grid of rebar and muttered, “This is a crime against uptime! They’ve got a million-dollar rig just sitting there. With a few lines of Python and a reinforcement-learning module, that backhoe could’ve been pouring concrete at 2AM last night, and that hole would’ve been dug, framed, and pressure-tested by dawn. Instead, they’re burning daylight—and ROI.”
“Just sitting there”, Pinter, 47, a regional logistics analyst, whose robotics experience was limited to programming his Roomba, but who once built an IKEA bookshelf with only 1 unused screw and has watched over a dozen YouTube videos & AI-related podcasts, elaborated for his unseen audience: “You’ve got millions in capital equipment depreciating. Why isn’t this automated? Robots don’t need Sundays off. Legacy humans must sleep, but rust doesn’t, and progress shouldn’t. 5 autonomous shifts will beat 1 union shift any day of the week. Even my cat gets more done!”
Sources say Pinter gestured at the silent bulldozer as though it might heed his TED-talk-level insight.""",
    """
Antichrist lectures are the hot new thing in Silicon Valley, but so far they’ve honestly been kind of disappointing. Some people are giving entire lecture series without even revealing who the Antichrist is! You’d expect that to be the bare minimum!

We can do better. Earlier this year, at the Manifest forecasting conference in Berkeley, I gave a presentation titled “Forecasting Transformative AI Using The Book Of Revelation”. Given the renewed interest in this topic, I repost it below, in written form, with slight edits. The first two-thirds, including the section on the Antichrist, is free. In deference to the injunction by prior researchers on this topic not to “cast one’s pearls before swine”, the last third (including the Whore of Babylon, the Battle of Armageddon, and the New Jerusalem) will be paywalled.

Thank you for attending. The Book of Revelation was written around 95 AD by St. John of Patmos. Most secular scholars interpret it as an allegorical description of events in John’s own time, especially the Roman persecution of the early church. But millennia of Christian commentators have treated it as a prophecy about some future cataclysm - most often during the commentator’s own era. In the 10th century, a renegade bishop declared Pope John XV to be the Antichrist. In the 19th century, the Russian Old Believers accused Napoleon of the same. In our own day, American evangelicals have proposed everyone from Saddam Hussein to Barack Obama.""",
    """In my 2019 post Too Much Dark Money In Almonds, I asked: why is there so little money in politics?

During the 2018 election, Americans - candidates, parties, PACs, and small donors like you - spent a combined $5 billion pushing their preferred candidates. Although that sounds like a lot of money, Americans spent $12 billion on almonds that same year. Why the imbalance? The oil industry has strong political opinions, and they make $500 billion per year. Do they really think electing oil-friendly politicians isn’t worth 2% of revenue?

We debated how this could be. Some of the discussion proved prescient - I asked if maybe Elon Musk should buy some kind of social media property. But we never found a good answer, and the implied question remained open: if some billionaire wanted to spend an actually relevant percent of his net worth on politics, could he just take over everything?

I recently talked to some Silicon Valley political consultants who updated me on the status of this issue: Marc Andreessen tried this in 2024 and it basically worked. Now he is trying it a second time, it will probably work again, and Marc Andreessen will probably own every politician twice over.""",
    """The following three things can’t all be true simultaneously:

Many Americans are fascists

Fascists are an acceptable target for political violence

Political violence in America is morally unacceptable (at the current time)

I thought about this while following the Twitter spat between Democratic hopeful Gavin Newsom and Trump advisor Stephen Miller. Newsom called Miller fascist; Miller accused this of being a call to violence which placed “a target” on him.

Miller is hardly sympathetic here - he’s called people fascist himself in the past, and later suggested Newsom should be arrested for his speech (if only there were a word to describe the sort of person who supports that kind of thing…)

Still, I found myself able to see things from both perspectives.""",
    """In 1917, three Portuguese children reported a vision of the Virgin Mary. She promised to return to them on the 13th of each month. On the sixth month - October 13th - she would perform a great miracle.

Rumors spread, and on the 13th of each month, crowds gathered to watch the children speak to an apparition that only they could see. Increasingly many of these pilgrims started reporting minor visions or miracles themselves. Anticipation for the great October miracle consumed the region, then the country.

On October 13, a crowd of about 70,000 people descended on the children’s home village of Fatima. At solar noon, the children made contact with the Virgin and said the great miracle was still on track. Then someone - accounts differ as to whether it was the children or a member of the crowd - pointed to the sky.

According to the ~150 eyewitness accounts that have come down to us, the clouds parted, and the pilgrims saw a strange pale sun (or sun-like object), painless to gaze upon. As they watched in wonder, it began to spin around and flash all the colors of the rainbow, drenching the trees and buildings and crowd with yellow, green, and purple light in sequence.""",
    """Something is off about this Bay Area House Party. There are . . . women.

“I’ve never seen a gender balance like this in the Bay Area,” you tell your host Chris. “Is this one of those fabled ratio parties?”

“No - have you heard of curtfishing? It’s the new male dating trend. You say in your Bumble profile that you’re a member of the Dissident Right who often attends parties with Curtis Yarvin. Then female journos ask you out in the hopes that you’ll bring them along and they can turn it into an article.”

“What happens when they realize Curtis Yarvin isn’t at the party?”

“Oh, everyone pools their money and hires someone to pretend to be Curtis. You can just do things. Today it’s Ramchandra.”""",
    """1. The Internet That Would Be
In July 1945, Vannevar Bush was riding high.

As Director of the Office of Scientific Research and Development, he’d won World War II. His proximity fuse intercepted hundreds of V-1s and destroyed thousands of tanks, carving a path for Allied forces through the French countryside. Back in 1942, he’d advocated to President Roosevelt the merits of Oppenheimer’s atomic bomb. Roosevelt and his congressional allies snuck hundreds of millions in covert funding to the OSRD’s planned projects in Oak Ridge and Los Alamos. Writing directly and secretively to Bush, a one-line memo in June expressed Roosevelt’s total confidence in his Director: “Do you have the money?”

Indeed he did. The warheads it bought would fall on Hiroshima and Nagasaki in mere weeks. The Germans had already given up; Victory in the Pacific was nigh. So Bush was thinking ahead.""",
    """Someone argues that Donald Trump threatens democracy, maybe because he’s asserting authority against the judiciary or the media or the NGOs. Someone else counterargues that it hardly seems undemocratic for someone to favor someone who won an election (the President) over other people who did not (the judiciary, the media). If anything, it seems undemocratic to allow the unelected people to continue to obstruct and harass elected leaders.

The most common response is to say that fine, democracy is about who wins votes, but we also like liberalism, liberalism is under threat, it’s too hard to talk about “liberalism” because in the US it sometimes means being left-wing, and so we use the related concept “democracy” as a stand-in. This is reasonable, and some accused-democracy-destroyers like Viktor Orban even accept it for themselves, calling their brand of government “illiberal democracy”.""",
    """If you’ve been following this blog for long, you probably know at least a bit about pharmaceutical research. You might know a bit about the sort of subtle measures pharmaceutical companies take to influence doctors’ prescribing habits, or how it takes billions of dollars on average to bring a new medication to market, or something about the perverse incentives which determine the FDA’s standards for accepting or rejecting a new drug. You might have some idea what kinds of hoops a company has to jump through to conduct actual research which meets legal guidelines for patient safety and autonomy.

You may be less familiar though, with how the sausage is actually made. How do pharmaceutical companies actually go through the process of testing a drug on human participants?

I’m going to be focusing here on a research subject’s view of what are known as Phase I clinical trials, the stage in which prospective drugs are tested for safety and tolerability. This is where researchers aim to answer questions like “Does this drug have any dangerous side effects?” “Through what pathways is it removed from a patient’s body?” and “Can we actually give people enough of this drug that it’s useful for anything?” This comes before the stage where researchers test how good a drug is at actually treating any sort of disease, when patients who’re suffering from the target ailments are given the option receive it as an experimental treatment. In Phase I clinical trials, the participants are healthy volunteers who’re participating in research for money. There are almost no cases in which volunteer participation is driven by motivations other than money, because the attitudes between research participants and clinicians overwhelmingly tend to be characterized by mutual guarded distrust. This distrust is baked into the process, both on a cultural level among the participants, and by the clinics’ own incentives.""",
    """AI psychosis (NYT, PsychologyToday) is an apparent phenomenon where people go crazy after talking to chatbots too much. There are some high-profile anecdotes, but still many unanswered questions. For example, how common is it really? Are the chatbots really driving people crazy, or just catching the attention of people who were crazy already? Isn’t psychosis supposed to be a biological disease? Wouldn’t that make chatbot-induced psychosis the same kind of category error as chatbot-induced diabetes?

I don’t have all the answers, so think of this post as an exploration of possible analogies and precedents rather than a strongly-held thesis. Also, I might have one answer - I think the yearly incidence of AI psychosis is somewhere around 1 in 10,000 (for a loose definition) to 1 in 100,000 (for a strict definition). I’ll talk about how I got those numbers at the end. But first:

I. Lenin Was A Mushroom
In the early 1990s, as the Soviet Union was collapsing, performance artist Sergey Kuryokhin presented a Daily Show style segment on a Russian talk show. He argued that Vladimir Lenin ate so many mushrooms that he eventually turned into a mushroom, and led the October Revolution while possessed by a sentient mushroom spirit.""",
    """You hold in your hands a compilation of two years of daily blog posts. In retrospect, I look back on that project and see a large number of things I did completely wrong. I’m fine with that. Looking back and not seeing a huge number of things I did wrong would mean that neither my writing nor my understanding had improved since 2009. Oops is the sound we make when we improve our beliefs and strategies; so to look back at a time and not see anything you did wrong means that you haven’t learned anything or changed your mind since then.

It was a mistake that I didn’t write my two years of blog posts with the intention of helping people do better in their everyday lives. I wrote it with the intention of helping people solve big, difficult, important problems, and I chose impressive-sounding, abstract problems as my examples.

In retrospect, this was the second-largest mistake in my approach. It ties in to the first-largest mistake in my writing, which was that I didn’t realize that the big problem in learning this valuable way of thinking was figuring out how to practice it, not knowing the theory. I didn’t realize that part was the priority; and regarding this I can only say “Oops” and “Duh.”""",
    """Imagine reaching into an urn that contains seventy white balls and thirty red ones, and plucking out ten mystery balls.

Perhaps three of the ten balls will be red, and you’ll correctly guess how many red balls total were in the urn. Or perhaps you’ll happen to grab four red balls, or some other number. Then you’ll probably get the total number wrong.

This random error is the cost of incomplete knowledge, and as errors go, it’s not so bad. Your estimates won’t be incorrect on average, and the more you learn, the smaller your error will tend to be.

On the other hand, suppose that the white balls are heavier, and sink to the bottom of the urn. Then your sample may be unrepresentative in a consistent direction.

That kind of error is called “statistical bias.” When your method of learning about the world is biased, learning more may not help. Acquiring more data can even consistently worsen a biased prediction.""",
    """Once upon a time, three groups of subjects were asked how much they would pay to save 2,000 / 20,000 / 200,000 migrating birds from drowning in uncovered oil ponds. The groups respectively answered $80, $78, and $88.1 This is scope insensitivity or scope neglect: the number of birds saved—the scope of the altruistic action—had little effect on willingness to pay.

Similar experiments showed that Toronto residents would pay little more to clean up all polluted lakes in Ontario than polluted lakes in a particular region of Ontario, or that residents of four western US states would pay only 28% more to protect all 57 wilderness areas in those states than to protect a single area.2 People visualize “a single exhausted bird, its feathers soaked in black oil, unable to escape.”3 This image, or prototype, calls forth some level of emotional arousal that is primarily responsible for willingness-to-pay—and the image is the same in all cases. As for scope, it gets tossed out the window—no human can visualize 2,000 birds at once, let alone 200,000. The usual finding is that exponential increases in scope create linear increases in willingness-to-pay—perhaps corresponding to the linear time for our eyes to glaze over the zeroes; this small amount of affect is added, not multiplied, with the prototype affect. This hypothesis is known as “valuation by prototype.”

An alternative hypothesis is “purchase of moral satisfaction.” People spend enough money to create a warm glow in themselves, a sense of having done their duty. The level of spending needed to purchase a warm glow depends on personality and financial situation, but it certainly has nothing to do with the number of birds.""",
    """I often use the metaphor that rationality is the martial art of mind. You don’t need huge, bulging muscles to learn martial arts—there’s a tendency toward more athletic people being more likely to learn martial arts, but that may be a matter of enjoyment as much as anything else. If you have a hand, with tendons and muscles in the appropriate places, then you can learn to make a fist.

Similarly, if you have a brain, with cortical and subcortical areas in the appropriate places, you might be able to learn to use it properly. If you’re a fast learner, you might learn faster—but the art of rationality isn’t about that; it’s about training brain machinery we all have in common. And where there are systematic errors human brains tend to make—like an insensitivity to scope—rationality is about fixing those mistakes, or finding work-arounds.

Alas, our minds respond less readily to our will than our hands. Our ability to control our muscles is evolutionarily ancient; our ability to reason about our own reasoning processes is a much more recent innovation. We shouldn’t be surprised, then, that muscles are easier to use than brains. But it is not wise to neglect the latter training because it is more difficult. It is not by bigger muscles that the human species rose to prominence upon Earth.""",
    """The availability heuristic is judging the frequency or probability of an event by the ease with which examples of the event come to mind.

A famous 1978 study by Lichtenstein, Slovic, Fischhoff, Layman, and Combs, “Judged Frequency of Lethal Events,” studied errors in quantifying the severity of risks, or judging which of two dangers occurred more frequently. Subjects thought that accidents caused about as many deaths as disease; thought that homicide was a more frequent cause of death than suicide. Actually, diseases cause about sixteen times as many deaths as accidents, and suicide is twice as frequent as homicide.

An obvious hypothesis to account for these skewed beliefs is that murders are more likely to be talked about than suicides—thus, someone is more likely to recall hearing about a murder than hearing about a suicide. Accidents are more dramatic than diseases—perhaps this makes people more likely to remember, or more likely to recall, an accident. In 1979, a followup study by Combs and Slovic showed that the skewed probability judgments correlated strongly (0.85 and 0.89) with skewed reporting frequencies in two newspapers. This doesn’t disentangle whether murders are more available to memory because they are more reported-on, or whether newspapers report more on murders because murders are more vivid (hence also more remembered). But either way, an availability bias is at work.""",
    """The availability heuristic is a cognitive shortcut humans use to reach conclusions; and where this shortcut reliably causes inaccurate conclusions, we can say that an availability bias is at work. Scope insensitivity is another example of a cognitive bias.

“Cognitive biases” are those obstacles to truth which are produced, not by the cost of information, nor by limited computing power, but by the shape of our own mental machinery. For example, our mental processes might be evolutionarily adapted to specifically believe some things that arent true, so that we could win political arguments in a tribal context. Or the mental machinery might be adapted not to particularly care whether something is true, such as when we feel the urge to believe what others believe to get along socially. Or the bias may be a side-effect of a useful reasoning heuristic. The availability heuristic is not itself a bias, but it gives rise to them; the machinery uses an algorithm (give things more evidential weight if they come to mind more readily) that does some good cognitive work but also produces systematic errors.

Our brains are doing something wrong, and after a lot of experimentation and/or heavy thinking, someone identifies the problem verbally and concretely; then we call it a “(cognitive) bias.” Not to be confused with the colloquial “that person is biased,” which just means “that person has a skewed or prejudiced attitude toward something.”""",
    """The conjunction fallacy is when humans assign a higher probability to a proposition of the form “A and B” than to one of the propositions “A” or “B” in isolation, even though it is a theorem that conjunctions are never likelier than their conjuncts. For example, in one experiment, 68% of the subjects ranked it more likely that “Reagan will provide federal support for unwed mothers and cut federal support to local governments” than that “Reagan will provide federal support for unwed mothers.”1

A long series of cleverly designed experiments, which weeded out alternative hypotheses and nailed down the standard interpretation, confirmed that conjunction fallacy occurs because we “substitute judgment of representativeness for judgment of probability.”2 By adding extra details, you can make an outcome seem more characteristic of the process that generates it. You can make it sound more plausible that Reagan will support unwed mothers, by adding the claim that Reagan will also cut support to local governments. The implausibility of one claim is compensated by the plausibility of the other; they “average out.”

Which is to say: Adding detail can make a scenario sound more plausible, even though the event necessarily becomes less probable.""",
    """I mean two things:

1. Epistemic rationality: systematically improving the accuracy of your beliefs.

2. Instrumental rationality: systematically achieving your values.

The first concept is simple enough. When you open your eyes and look at the room around you, you’ll locate your laptop in relation to the table, and you’ll locate a bookcase in relation to the wall. If something goes wrong with your eyes, or your brain, then your mental model might say there’s a bookcase where no bookcase exists, and when you go over to get a book, you’ll be disappointed.

This is what it’s like to have a false belief, a map of the world that doesn’t correspond to the territory. Epistemic rationality is about building accurate maps instead. This correspondence between belief and reality is commonly called “truth,” and I’m happy to call it that.1

Instrumental rationality, on the other hand, is about steering reality—sending the future where you want it to go. It’s the art of choosing actions that lead to outcomes ranked higher in your preferences. I sometimes call this “winning.”""",
    """The Denver International Airport opened 16 months late, at a cost overrun of $2 billion.1

The Eurofighter Typhoon, a joint defense project of several European countries, was delivered 54 months late at a cost of $19 billion instead of $7 billion.

The Sydney Opera House may be the most legendary construction overrun of all time, originally estimated to be completed in 1963 for $7 million, and finally completed in 1973 for $102 million.2

Are these isolated disasters brought to our attention by selective availability? Are they symptoms of bureaucracy or government incentive failures? Yes, very probably. But there’s also a corresponding cognitive bias, replicated in experiments with individual planners.

Buehler et al. asked their students for estimates of when they (the students) thought they would complete their personal academic projects.3 Specifically, the researchers asked for estimated times by which the students thought it was 50%, 75%, and 99% probable their personal projects would be done. Would you care to guess how many students finished on or before their estimated 50%, 75%, and 99% probability levels?""",
    """The goal of instrumental rationality mostly speaks for itself. Some commenters have wondered, on the other hand, why rationalists care about truth. Which invites a few different answers, depending on who you ask; and these different answers have differing characters, which can shape the search for truth in different ways.

You might hold the view that pursuing truth is inherently noble, important, and worthwhile. In which case your priorities will be determined by your ideals about which truths are most important, or about when truthseeking is most virtuous.

This motivation tends to have a moral character to it. If you think it your duty to look behind the curtain, you are a lot more likely to believe that someone else should look behind the curtain too, or castigate them if they deliberately close their eyes.

I tend to be suspicious of morality as a motivation for rationality, not because I reject the moral ideal, but because it invites certain kinds of trouble. It is too easy to acquire, as learned moral duties, modes of thinking that are dreadful missteps in the dance.

Consider Spock, the naive archetype of rationality. Spock's affect is always set to “calm,” even when wildly inappropriate. He often gives many significant digits for probabilities that are grossly uncalibrated.1 Yet this popular image is how many people conceive of the duty to be “rational”—small wonder that they do not embrace it wholeheartedly.""",
    """Since curiosity is an emotion, I suspect that some people will object to treating curiosity as a part of rationality. A popular belief about “rationality” is that rationality opposes all emotion—that all our sadness and all our joy are automatically anti-logical by virtue of being feelings. Yet strangely enough, I can’t find any theorem of probability theory which proves that I should appear ice-cold and expressionless.

When people think of “emotion” and “rationality” as opposed, I suspect that they are really thinking of System 1 and System 2—fast perceptual judgments versus slow deliberative judgments. System 2’s deliberative judgments aren’t always true, and System 1’s perceptual judgments aren’t always false; so it is very important to distinguish that dichotomy from “rationality.” Both systems can serve the goal of truth, or defeat it, depending on how they are used.

For my part, I label an emotion as “not rational” if it rests on mistaken beliefs, or rather, on mistake-producing epistemic conduct. “If the iron approaches your face, and you believe it is hot, and it is cool, the Way opposes your fear. If the iron approaches your face, and you believe it is cool, and it is hot, the Way opposes your calm.” Conversely, an emotion that is evoked by correct beliefs or truth-conducive thinking is a “rational emotion”; and this has the advantage of letting us regard calm as an emotional state, rather than a privileged default.""",
    """Light leaves the Sun and strikes your shoelaces and bounces off; some photons enter the pupils of your eyes and strike your retina; the energy of the photons triggers neural impulses; the neural impulses are transmitted to the visual-processing areas of the brain; and there the optical information is processed and reconstructed into a 3D model that is recognized as an untied shoelace; and so you believe that your shoelaces are untied.

Here is the secret of deliberate rationality—this whole process is not magic, and you can understand it. You can understand how you see your shoelaces. You can think about which sort of thinking processes will create beliefs which mirror reality, and which thinking processes will not.

Mice can see, but they can’t understand seeing. You can understand seeing, and because of that, you can do things that mice cannot do. Take a moment to marvel at this, for it is indeed marvelous.

Mice see, but they don’t know they have visual cortexes, so they can’t correct for optical illusions. A mouse lives in a mental world that includes cats, holes, cheese and mousetraps—but not mouse brains. Their camera does not take pictures of its own lens. But we, as humans, can look at a seemingly bizarre image, and realize that part of what we’re seeing is the lens itself. You don’t always have to believe your own eyes, but you have to realize that you have eyes—you must have distinct mental buckets for the map and the territory, for the senses and reality. Lest you think this a trivial ability, remember how rare it is in the animal kingdom.""",
    """Thus begins the ancient parable:

If a tree falls in a forest and no one hears it, does it make a sound? One says, “Yes it does, for it makes vibrations in the air.” Another says, “No it does not, for there is no auditory processing in any brain.”

If there’s a foundational skill in the martial art of rationality, a mental stance on which all other technique rests, it might be this one: the ability to spot, inside your own head, psychological signs that you have a mental map of something, and signs that you don’t.

Suppose that, after a tree falls, the two arguers walk into the forest together. Will one expect to see the tree fallen to the right, and the other expect to see the tree fallen to the left? Suppose that before the tree falls, the two leave a sound recorder next to the tree. Would one, playing back the recorder, expect to hear something different from the other? Suppose they attach an electroencephalograph to any brain in the world; would one expect to see a different trace than the other?

Though the two argue, one saying “No,” and the other saying “Yes,” they do not anticipate any different experiences. The two think they have different models of the world, but they have no difference with respect to what they expect will happen to them; their maps of the world do not diverge in any sensory detail.""",
    """In the time of the Roman Empire, civic life was divided between the Blue and Green factions. The Blues and the Greens murdered each other in single combats, in ambushes, in group battles, in riots. Procopius said of the warring factions: “So there grows up in them against their fellow men a hostility which has no cause, and at no time does it cease or disappear, for it gives place neither to the ties of marriage nor of relationship nor of friendship, and the case is the same even though those who differ with respect to these colors be brothers or any other kin.”1 Edward Gibbon wrote: “The support of a faction became necessary to every candidate for civil or ecclesiastical honors.”2

Who were the Blues and the Greens? They were sports fans—the partisans of the blue and green chariot-racing teams.

Imagine a future society that flees into a vast underground network of caverns and seals the entrances. We shall not specify whether they flee disease, war, or radiation; we shall suppose the first Undergrounders manage to grow food, find water, recycle air, make light, and survive, and that their descendants thrive and eventually form cities. Of the world above, there are only legends written on scraps of paper; and one of these scraps of paper describes the sky, a vast open space of air above a great unbounded floor. The sky is cerulean in color, and contains strange floating objects like enormous tufts of white cotton. But the meaning of the word “cerulean” is controversial; some say that it refers to the color known as “blue,” and others that it refers to the color known as “green.”""",
    """Carl Sagan once told a parable of someone who comes to us and claims: “There is a dragon in my garage.” Fascinating! We reply that we wish to see this dragon—let us set out at once for the garage! “But wait,” the claimant says to us, “it is an invisible dragon.”

Now as Sagan points out, this doesn’t make the hypothesis unfalsifiable. Perhaps we go to the claimant’s garage, and although we see no dragon, we hear heavy breathing from no visible source; footprints mysteriously appear on the ground; and instruments show that something in the garage is consuming oxygen and breathing out carbon dioxide.

But now suppose that we say to the claimant, “Okay, we’ll visit the garage and see if we can hear heavy breathing,” and the claimant quickly says no, it’s an inaudible dragon. We propose to measure carbon dioxide in the air, and the claimant says the dragon does not breathe. We propose to toss a bag of flour into the air to see if it outlines an invisible dragon, and the claimant immediately says, “The dragon is permeable to flour.”""",
    """The earliest account I know of a scientific experiment is, ironically, the story of Elijah and the priests of Baal.

The people of Israel are wavering between Jehovah and Baal, so Elijah announces that he will conduct an experiment to settle it—quite a novel concept in those days! The priests of Baal will place their bull on an altar, and Elijah will place Jehovah’s bull on an altar, but neither will be allowed to start the fire; whichever God is real will call down fire on His sacrifice. The priests of Baal serve as control group for Elijah—the same wooden fuel, the same bull, and the same priests making invocations, but to a false god. Then Elijah pours water on his altar—ruining the experimental symmetry, but this was back in the early days—to signify deliberate acceptance of the burden of proof, like needing a 0.05 significance level. The fire comes down on Elijah’s altar, which is the experimental observation. The watching people of Israel shout “The Lord is God!”—peer review.

And then the people haul the 450 priests of Baal down to the river Kishon and slit their throats. This is stern, but necessary. You must firmly discard the falsified hypothesis, and do so swiftly, before it can generate excuses to protect itself. If the priests of Baal are allowed to survive, they will start babbling about how religion is a separate magisterium which can be neither proven nor disproven.""",
    """I once attended a panel on the topic, “Are science and religion compatible?” One of the women on the panel, a pagan, held forth interminably upon how she believed that the Earth had been created when a giant primordial cow was born into the primordial abyss, who licked a primordial god into existence, whose descendants killed a primordial giant and used its corpse to create the Earth, etc. The tale was long, and detailed, and more absurd than the Earth being supported on the back of a giant turtle. And the speaker clearly knew enough science to know this.

I still find myself struggling for words to describe what I saw as this woman spoke. She spoke with . . . pride? Self-satisfaction? A deliberate flaunting of herself?

The woman went on describing her creation myth for what seemed like forever, but was probably only five minutes. That strange pride/satisfaction/flaunting clearly had something to do with her knowing that her beliefs were scientifically outrageous. And it wasn’t that she hated science; as a panelist she professed that religion and science were compatible. She even talked about how it was quite understandable that the Vikings talked about a primordial abyss, given the land in which they lived—explained away her own religion!—and yet nonetheless insisted this was what she “believed,” said with peculiar satisfaction.""",
    """I have so far distinguished between belief as anticipation-controller, belief in belief, professing and cheering.  Of these, we might call anticipation-controlling beliefs "proper beliefs" and the other forms "improper belief". Proper belief can be wrong or irrational, as when someone genuinely anticipates that prayer will cure their sick baby. But the other forms are arguably “not belief at all.”

Yet another form of improper belief is belief as group identification—as a way of belonging. Robin Hanson uses the excellent metaphor of wearing unusual clothing, a group uniform like a priest’s vestments or a Jewish skullcap, and so I will call this “belief as attire.”

In terms of humanly realistic psychology, the Muslims who flew planes into the World Trade Center undoubtedly saw themselves as heroes defending truth, justice, and the Islamic Way from hideous alien monsters a la the movie Independence Day. Only a very inexperienced nerd, the sort of nerd who has no idea how non-nerds see the world, would say this out loud in an Alabama bar. It is not an American thing to say. The American thing to say is that the terrorists “hate our freedom” and that flying a plane into a building is a “cowardly act.” You cannot say the phrases “heroic self-sacrifice” and “suicide bomber” in the same sentence, even for the sake of accurately describing how the Enemy sees the world. The very concept of the courage and altruism of a suicide bomber is Enemy attire—you can tell, because the Enemy talks about it. The cowardice and sociopathy of a suicide bomber is American attire. There are no quote marks you can use to talk about how the Enemy sees the world; it would be like dressing up as a Nazi for Halloween.""",
    """Belief is quantitative, and just as it is possible to make overconfident assertions relative to ones anticipations, it is possible to make under confident assertions relative to ones anticipations. One can wear the attire of uncertainty, or profess an agnosticism that isn’t really there. Here, I'll single out a special case of improper uncertainty: the display of neutrality or suspended judgment in order to signal maturity, impartiality, or a superior vantage point.

An example would be the case of my parents, who respond to theological questions like “Why does ancient Egypt, which had good records on many other matters, lack any records of Jews having ever been there?” with “Oh, when I was your age, I also used to ask that sort of question, but now I’ve grown out of it.”

Another example would be the principal who, faced with two children who were caught fighting on the playground, sternly says: “It doesn’t matter who started the fight, it only matters who ends it.” Of course it matters who started the fight. The principal may not have access to good information about this critical fact, but if so, the principal should say so, not dismiss the importance of who threw the first punch. Let a parent try punching the principal, and we’ll see how far “It doesn’t matter who started it” gets in front of a judge. But to adults it is just inconvenient that children fight, and it matters not at all to their convenience which child started it. It is only convenient that the fight end as rapidly as possible.""",
    """At the Singularity Summit 2007, one of the speakers called for democratic, multinational development of artificial intelligence. So I stepped up to the microphone and asked:

Suppose that a group of democratic republics form a consortium to develop AI, and there’s a lot of politicking during the process—some interest groups have unusually large influence, others get shafted—in other words, the result looks just like the products of modern democracies. Alternatively, suppose a group of rebel nerds develops an AI in their basement, and instructs the AI to poll everyone in the world—dropping cellphones to anyone who doesn’t have them—and do whatever the majority says. Which of these do you think is more “democratic,” and would you feel safe with either?

I wanted to find out whether he believed in the pragmatic adequacy of the democratic political process, or if he believed in the moral rightness of voting. But the speaker replied:

The first scenario sounds like an editorial in Reason magazine, and the second sounds like a Hollywood movie plot.""",
    """Will bond yields go up, or down, or remain the same? If you’re a TV pundit and your job is to explain the outcome after the fact, then there’s no reason to worry. No matter which of the three possibilities comes true, you’ll be able to explain why the outcome perfectly fits your pet market theory. There’s no reason to think of these three possibilities as somehow opposed to one another, as exclusive, because you’ll get full marks for punditry no matter which outcome occurs.

But wait! Suppose you’re a novice TV pundit, and you aren’t experienced enough to make up plausible explanations on the spot. You need to prepare remarks in advance for tomorrow’s broadcast, and you have limited time to prepare. In this case, it would be helpful to know which outcome will actually occur—whether bond yields will go up, down, or remain the same—because then you would only need to prepare one set of excuses.

Alas, no one can possibly foresee the future. What are you to do? You certainly can’t use “probabilities.” We all know from school that “probabilities” are little numbers that appear next to a word problem, and there aren’t any little numbers here. Worse, you feel uncertain. You don’t remember feeling uncertain while you were manipulating the little numbers in word problems. College classes teaching math are nice clean places, so math can’t apply to life situations that aren’t nice and clean. You wouldn’t want to inappropriately transfer thinking skills from one context to another. Clearly, this is not a matter for “probabilities.”""",
    """Walking along the street, your shoelaces come untied. Shortly thereafter, for some odd reason, you start believing your shoelaces are untied. Light leaves the Sun and strikes your shoelaces and bounces off; some photons enter the pupils of your eyes and strike your retina; the energy of the photons triggers neural impulses; the neural impulses are transmitted to the visual-processing areas of the brain; and there the optical information is processed and reconstructed into a 3D model that is recognized as an untied shoelace. There is a sequence of events, a chain of cause and effect, within the world and your brain, by which you end up believing what you believe. The final outcome of the process is a state of mind which mirrors the state of your actual shoelaces.

What is evidence? It is an event entangled, by links of cause and effect, with whatever you want to know about. If the target of your inquiry is your shoelaces, for example, then the light entering your pupils is evidence entangled with your shoelaces. This should not be confused with the technical sense of “entanglement” used in physics—here I’m just talking about “entanglement” in the sense of two things that end up in correlated states because of the links of cause and effect between them.

Not every influence creates the kind of “entanglement” required for evidence. It’s no help to have a machine that beeps when you enter winning lottery numbers, if the machine also beeps when you enter losing lottery numbers. The light reflected from your shoes would not be useful evidence about your shoelaces, if the photons ended up in the same physical state whether your shoelaces were tied or untied.""",
    """Suppose that your good friend, the police commissioner, tells you in strictest confidence that the crime kingpin of your city is Wulky Wilkinsen. As a rationalist, are you licensed to believe this statement? Put it this way: if you go ahead and insult Wulky, I’d call you foolhardy. Since it is prudent to act as if Wulky has a substantially higher-than-default probability of being a crime boss, the police commissioner’s statement must have been strong Bayesian evidence.

Our legal system will not imprison Wulky on the basis of the police commissioner’s statement. It is not admissible as legal evidence. Maybe if you locked up every person accused of being a crime boss by a police commissioner, you’d initially catch a lot of crime bosses, and relatively few people the commissioner just didn’t like. But unrestrained power attracts corruption like honey attracts flies: over time, you’d catch fewer and fewer real crime bosses (who would go to greater lengths to ensure anonymity), and more and more innocent victims.

This does not mean that the police commissioner’s statement is not rational evidence. It still has a lopsided likelihood ratio, and you’d still be a fool to insult Wulky. But on a social level, in pursuit of a social goal, we deliberately define “legal evidence” to include only particular kinds of evidence, such as the police commissioner’s own observations on the night of April 4th. All legal evidence should ideally be rational evidence, but not the other way around. We impose special, strong, additional standards before we anoint rational evidence as “legal evidence.”""",
    """Previously, I defined evidence as “an event entangled, by links of cause and effect, with whatever you want to know about,” and entangled as “happening differently for different possible states of the target.” So how much entanglement—how much rational evidence—is required to support a belief?

Let’s start with a question simple enough to be mathematical: How hard would you have to entangle yourself with the lottery in order to win? Suppose there are seventy balls, drawn without replacement, and six numbers to match for the win. Then there are 131,115,985 possible winning combinations, hence a randomly selected ticket would have a 1/131,115,985 probability of winning (0.0000007%). To win the lottery, you would need evidence selective enough to visibly favor one combination over 131,115,984 alternatives.

Suppose there are some tests you can perform which discriminate, probabilistically, between winning and losing lottery numbers. For example, you can punch a combination into a little black box that always beeps if the combination is the winner, and has only a 1/4 (25%) chance of beeping if the combination is wrong. In Bayesian terms, we would say the likelihood ratio is 4 to 1. This means that the box is 4 times as likely to beep when we punch in a correct combination, compared to how likely it is to beep for an incorrect combination.""",
    """In 1919, Sir Arthur Eddington led expeditions to Brazil and to the island of Principe, aiming to observe solar eclipses and thereby test an experimental prediction of Einstein’s novel theory of General Relativity. A journalist asked Einstein what he would do if Eddington’s observations failed to match his theory. Einstein famously replied: “Then I would feel sorry for the good Lord. The theory is correct.”

It seems like a rather foolhardy statement, defying the trope of Traditional Rationality that experiment above all is sovereign. Einstein seems possessed of an arrogance so great that he would refuse to bend his neck and submit to Nature’s answer, as scientists must do. Who can know that the theory is correct, in advance of experimental test?

Of course, Einstein did turn out to be right. I try to avoid criticizing people when they are right. If they genuinely deserve criticism, I will not need to wait long for an occasion where they are wrong.

And Einstein may not have been quite so foolhardy as he sounded . . .""",
    """The more complex an explanation is, the more evidence you need just to find it in belief-space. (In Traditional Rationality this is often phrased misleadingly, as “The more complex a proposition is, the more evidence is required to argue for it.”) How can we measure the complexity of an explanation? How can we determine how much evidence is required?

Occam’s Razor is often phrased as “The simplest explanation that fits the facts.” Robert Heinlein replied that the simplest explanation is “The lady down the street is a witch; she did it.”

One observes that the length of an English sentence is not a good way to measure “complexity.” And “fitting” the facts by merely failing to prohibit them is insufficient.

Why, exactly, is the length of an English sentence a poor measure of complexity? Because when you speak a sentence aloud, you are using labels for concepts that the listener shares—the receiver has already stored the complexity in them. Suppose we abbreviated Heinlein’s whole sentence as “Tldtsiawsdi!” so that the entire explanation can be conveyed in one word; better yet, we’ll give it a short arbitrary label like “Fnord!” Does this reduce the complexity? No, because you have to tell the listener in advance that “Tldtsiawsdi!” stands for “The lady down the street is a witch; she did it.” “Witch,” itself, is a label for some extraordinary assertions—just because we all know what it means doesn’t mean the concept is simple.""",
    """The following happened to me in an IRC chatroom, long enough ago that I was still hanging around in IRC chatrooms. Time has fuzzed the memory and my report may be imprecise.

So there I was, in an IRC chatroom, when someone reports that a friend of his needs medical advice. His friend says that he’s been having sudden chest pains, so he called an ambulance, and the ambulance showed up, but the paramedics told him it was nothing, and left, and now the chest pains are getting worse. What should his friend do?

I was confused by this story. I remembered reading about homeless people in New York who would call ambulances just to be taken someplace warm, and how the paramedics always had to take them to the emergency room, even on the 27th iteration. Because if they didn’t, the ambulance company could be sued for lots and lots of money. Likewise, emergency rooms are legally obligated to treat anyone, regardless of ability to pay.1 So I didn’t quite understand how the described events could have happened. Anyone reporting sudden chest pains should have been hauled off by an ambulance instantly.

And this is where I fell down as a rationalist. I remembered several occasions where my doctor would completely fail to panic at the report of symptoms that seemed, to me, very alarming. And the Medical Establishment was always right. Every single time. I had chest pains myself, at one point, and the doctor patiently explained to me that I was describing chest muscle pain, not a heart attack. So I said into the IRC channel, “Well, if the paramedics told your friend it was nothing, it must really be nothing—they’d have hauled him off if there was the tiniest chance of serious trouble.”""",
    """Consider Warren’s argument from a Bayesian perspective. When we see evidence, hypotheses that assigned a higher likelihood to that evidence gain probability, at the expense of hypotheses that assigned a lower likelihood to the evidence. This is a phenomenon of relative likelihoods and relative probabilities. You can assign a high likelihood to the evidence and still lose probability mass to some other hypothesis, if that other hypothesis assigns a likelihood that is even higher.

Warren seems to be arguing that, given that we see no sabotage, this confirms that a Fifth Column exists. You could argue that a Fifth Column might delay its sabotage. But the likelihood is still higher that the absence of a Fifth Column would perform an absence of sabotage.

Let E stand for the observation of sabotage, and ¬E for the observation of no sabotage. The symbol H1 stands for the hypothesis of a Japanese-American Fifth Column, and H2 for the hypothesis that no Fifth Column exists. The conditional probability P(E | H), or “E given H,” is how confidently we’d expect to see the evidence E if we assumed the hypothesis H were true.""",
    """Friedrich Spee von Langenfeld, a priest who heard the confessions of condemned witches, wrote in 1631 the Cautio Criminalis (“prudence in criminal cases”), in which he bitingly described the decision tree for condemning accused witches: If the witch had led an evil and improper life, she was guilty; if she had led a good and proper life, this too was a proof, for witches dissemble and try to appear especially virtuous. After the woman was put in prison: if she was afraid, this proved her guilt; if she was not afraid, this proved her guilt, for witches characteristically pretend innocence and wear a bold front. Or on hearing of a denunciation of witchcraft against her, she might seek flight or remain; if she ran, that proved her guilt; if she remained, the devil had detained her so she could not get away.

Spee acted as confessor to many witches; he was thus in a position to observe every branch of the accusation tree, that no matter what the accused witch said or did, it was held as proof against her. In any individual case, you would only hear one branch of the dilemma. It is for this reason that scientists write down their experimental predictions in advance.

But you can’t have it both ways —as a matter of probability theory, not mere fairness. The rule that “absence of evidence is evidence of absence” is a special case of a more general law, which I would name Conservation of Expected Evidence: the expectation of the posterior probability, after viewing the evidence, must equal the prior probability.""",
    """This essay is closely based on an excerpt from Meyers’s Exploring Social Psychology; the excerpt is worth reading in its entirety.

Cullen Murphy, editor of The Atlantic, said that the social sciences turn up “no ideas or conclusions that can’t be found in [any] encyclopedia of quotations . . . Day after day social scientists go out into the world. Day after day they discover that people’s behavior is pretty much what you’d expect.”

Of course, the “expectation” is all hindsight. (Hindsight bias: Subjects who know the actual answer to a question assign much higher probabilities they “would have” guessed for that answer, compared to subjects who must guess without knowing the answer.)

The historian Arthur Schlesinger, Jr. dismissed scientific studies of World War II soldiers’ experiences as “ponderous demonstrations” of common sense.""",
    """In hindsight bias, people who know the outcome of a situation believe the outcome should have been easy to predict in advance. Knowing the outcome, we reinterpret the situation in light of that outcome. Even when warned, we can’t de-interpret to empathize with someone who doesn’t know what we know.

Closely related is the illusion of transparency: We always know what we mean by our words, and so we expect others to know it too. Reading our own writing, the intended interpretation falls easily into place, guided by our knowledge of what we really meant. It’s hard to empathize with someone who must interpret blindly, guided only by the words.

June recommends a restaurant to Mark; Mark dines there and discovers (a) unimpressive food and mediocre service or (b) delicious food and impeccable service. Then Mark leaves the following message on June’s answering machine: “June, I just finished dinner at the restaurant you recommended, and I must say, it was marvelous, just marvelous.” Keysar (1994) presented a group of subjects with scenario (a), and 59% thought that Mark’s message was sarcastic and that Jane would perceive the sarcasm.1 Among other subjects, told scenario (b), only 3% thought that Jane would perceive Mark’s message as sarcastic. Keysar and Barr (2002) seem to indicate that an actual voice message was played back to the subjects.2 Keysar (1998) showed that if subjects were told that the restaurant was horrible but that Mark wanted to conceal his response, they believed June would not perceive sarcasm in the (same) message:3""",
    """Homo sapiens’s environment of evolutionary adaptedness (a.k.a. EEA or “ancestral environment”) consisted of hunter-gatherer bands of at most 200 people, with no writing. All inherited knowledge was passed down by speech and memory.

In a world like that, all background knowledge is universal knowledge. All information not strictly private is public, period.

In the ancestral environment, you were unlikely to end up more than one inferential step away from anyone else. When you discover a new oasis, you don’t have to explain to your fellow tribe members what an oasis is, or why it’s a good idea to drink water, or how to walk. Only you know where the oasis lies; this is private knowledge. But everyone has the background to understand your description of the oasis, the concepts needed to think about water; this is universal knowledge. When you explain things in an ancestral environment, you almost never have to explain your concepts. At most you have to explain one new concept, not two or more simultaneously.

In the ancestral environment there were no abstract disciplines with vast bodies of carefully gathered evidence generalized into elegant theories transmitted by written books whose conclusions are a hundred inferential steps removed from universally shared background premises.""",
    """Once upon a time, there was an instructor who taught physics students. One day the instructor called them into the classroom and showed them a wide, square plate of metal, next to a hot radiator. The students each put their hand on the plate and found the side next to the radiator cool, and the distant side warm. And the instructor said, Why do you think this happens? Some students guessed convection of air currents, and others guessed strange metals in the plate. They devised many creative explanations, none stooping so low as to say “I don’t know” or “This seems impossible.”

And the answer was that before the students entered the room, the instructor turned the plate around.1

Consider the student who frantically stammers, “Eh, maybe because of the heat conduction and so?” I ask: Is this answer a proper belief? The words are easily enough professed—said in a loud, emphatic voice. But do the words actually control anticipation?

Ponder that innocent little phrase, “because of,” which comes before “heat conduction.” Ponder some of the other things we could put after it. We could say, for example, “Because of phlogiston,” or “Because of magic.”""",
    """When I was young, I read popular physics books such as Richard Feynman’s QED: The Strange Theory of Light and Matter. I knew that light was waves, sound was waves, matter was waves. I took pride in my scientific literacy, when I was nine years old.

When I was older, and I began to read the Feynman Lectures on Physics, I ran across a gem called “the wave equation.” I could follow the equation’s derivation, but, looking back, I couldn’t see its truth at a glance. So I thought about the wave equation for three days, on and off, until I saw that it was embarrassingly obvious. And when I finally understood, I realized that the whole time I had accepted the honest assurance of physicists that light was waves, sound was waves, matter was waves, I had not had the vaguest idea of what the word “wave” meant to a physicist.

There is an instinctive tendency to think that if a physicist says “light is made of waves,” and the teacher says “What is light made of?” and the student says “Waves!”, then the student has made a true statement. That’s only fair, right? We accept “waves” as a correct answer from the physicist; wouldn’t it be unfair to reject it from the student? Surely, the answer “Waves!” is either true or false, right?""",
    """The preview for the X-Men movie has a voice-over saying: “In every human being . . . there is the genetic code . . . for mutation.” Apparently you can acquire all sorts of neat abilities by mutation. The mutant Storm, for example, has the ability to throw lightning bolts.

I beg you, dear reader, to consider the biological machinery necessary to generate electricity; the biological adaptations necessary to avoid being harmed by electricity; and the cognitive circuitry required for finely tuned control of lightning bolts. If we actually observed any organism acquiring these abilities in one generation, as the result of mutation, it would outright falsify the neo-Darwinian model of natural selection. It would be worse than finding rabbit fossils in the pre-Cambrian. If evolutionary theory could actually stretch to cover Storm, it would be able to explain anything, and we all know what that would imply.

The X-Men comics use terms like “evolution,” “mutation,” and “genetic code,” purely to place themselves in what they conceive to be the literary genre of science. The part that scares me is wondering how many people, especially in the media, understand science only as a literary genre.""",
    """Phlogiston was the eighteenth century’s answer to the Elemental Fire of the Greek alchemists. Ignite wood, and let it burn. What is the orangey-bright “fire” stuff? Why does the wood transform into ash? To both questions, the eighteenth-century chemists answered, “phlogiston.”

. . . and that was it, you see, that was their answer: “Phlogiston.”

Phlogiston escaped from burning substances as visible fire. As the phlogiston escaped, the burning substances lost phlogiston and so became ash, the “true material.” Flames in enclosed containers went out because the air became saturated with phlogiston, and so could not hold any more. Charcoal left little residue upon burning because it was nearly pure phlogiston.

Of course, one didn’t use phlogiston theory to predict the outcome of a chemical transformation. You looked at the result first, then you used phlogiston theory to explain it. It’s not that phlogiston theorists predicted a flame would extinguish in a closed container; rather they lit a flame in a container, watched it go out, and then said, “The air must have become saturated with phlogiston.” You couldn’t even use phlogiston theory to say what you ought not to see; it could explain everything.""",
    """Consider the seeming paradox of the First Cause. Science has traced events back to the Big Bang, but why did the Big Bang happen? It’s all well and good to say that the zero of time begins at the Big Bang—that there is nothing before the Big Bang in the ordinary flow of minutes and hours. But saying this presumes our physical law, which itself appears highly structured; it calls out for explanation. Where did the physical laws come from? You could say that we’re all a computer simulation, but then the computer simulation is running on some other world’s laws of physics—where did those laws of physics come from?

At this point, some people say, “God!”

What could possibly make anyone, even a highly religious person, think this even helped answer the paradox of the First Cause? Why wouldn’t you automatically ask, “Where did God come from?” Saying “God is uncaused” or “God created Himself” leaves us in exactly the same position as “Time began with the Big Bang.” We just ask why the whole metasystem exists in the first place, or why some events but not others are allowed to be uncaused.""",
    """Imagine looking at your hand, and knowing nothing of cells, nothing of biochemistry, nothing of DNA. You’ve learned some anatomy from dissection, so you know your hand contains muscles; but you don’t know why muscles move instead of lying there like clay. Your hand is just . . . stuff . . . and for some reason it moves under your direction. Is this not magic?

This was the theory of vitalism ; that the mysterious difference between living matter and non-living matter was explained by an Élan vital or vis vitalis. Élan vital infused living matter and caused it to move as consciously directed. Élan vital participated in chemical transformations which no mere non-living particles could undergo—Wöhler’s later synthesis of urea, a component of urine, was a major blow to the vitalistic theory because it showed that mere chemistry could duplicate a product of biology.

Calling “Élan vital” an explanation, even a fake explanation like phlogiston, is probably giving it too much credit. It functioned primarily as a curiosity-stopper. You said “Why?” and the answer was “Élan vital!”""",
    """The failures of phlogiston and vitalism are historical hindsight. Dare I step out on a limb, and name some current theory which I deem analogously flawed?

I name emergence or emergent phenomena—usually defined as the study of systems whose high-level behaviors arise or “emerge” from the interaction of many low-level elements. (Wikipedia: “The way complex systems and patterns arise out of a multiplicity of relatively simple interactions.”)

Taken literally, that description fits every phenomenon in our universe above the level of individual quarks, which is part of the problem. Imagine pointing to a market crash and saying “It’s not a quark!” Does that feel like an explanation? No? Then neither should saying “It’s an emergent phenomenon!”

It’s the noun “emergence” that I protest, rather than the verb “emerges from.” There’s nothing wrong with saying “X emerges from Y,” where Y is some specific, detailed model with internal moving parts. “Arises from” is another legitimate phrase that means exactly the same thing. Gravity arises from the curvature of spacetime, according to the specific mathematical model of General Relativity. Chemistry arises from interactions between atoms, according to the specific model of quantum electrodynamics.""",
    """This cognitive phenomenon is usually lumped in with “confirmation bias.” However, it seems to me that the phenomenon of trying to test positive rather than negative examples, ought to be distinguished from the phenomenon of trying to preserve the belief you started with. “Positive bias” is sometimes used as a synonym for “confirmation bias,” and fits this particular flaw much better.

It once seemed that phlogiston theory could explain a flame going out in an enclosed box (the air became saturated with phlogiston and no more could be released). But phlogiston theory could just as well have explained the flame not going out. To notice this, you have to search for negative examples instead of positive examples, look into zero instead of one; which goes against the grain of what experiment has shown to be human instinct.

For by instinct, we human beings only live in half the world.""",
    """Many psychological experiments were conducted in the late 1950s and early 1960s in which subjects were asked to predict the outcome of an event that had a random component but yet had base-rate predictability—for example, subjects were asked to predict whether the next card the experimenter turned over would be red or blue in a context in which 70% of the cards were blue, but in which the sequence of red and blue cards was totally random.

In such a situation, the strategy that will yield the highest proportion of success is to predict the more common event. For example, if 70% of the cards are blue, then predicting blue on every trial yields a 70% success rate.

What subjects tended to do instead, however, was match probabilities—that is, predict the more probable event with the relative frequency with which it occurred. For example, subjects tended to predict 70% of the time that the blue card would occur and 30% of the time that the red card would occur. Such a strategy yields a 58% success rate, because the subjects are correct 70% of the time when the blue card occurs (which happens with probability .70) and 30% of the time when the red card occurs (which happens with probability .30); (.70×.70) + (.30×.30) = .58.""",
    """It is said that parents do all the things they tell their children not to do, which is how they know not to do them.

Long ago, in the unthinkably distant past, I was a devoted Traditional Rationalist, conceiving myself skilled according to that kind, yet I knew not the Way of Bayes. When the young Eliezer was confronted with a mysterious-seeming question, the precepts of Traditional Rationality did not stop him from devising a Mysterious Answer. It is, by far, the most embarrassing mistake I made in my life, and I still wince to think of it.

What was my mysterious answer to a mysterious question? This I will not describe, for it would be a long tale and complicated. I was young, and a mere Traditional Rationalist who knew not the teachings of Tversky and Kahneman. I knew about Occam’s Razor, but not the conjunction fallacy. I thought I could get away with thinking complicated thoughts myself, in the literary style of the complicated thoughts I read in science books, not realizing that correct complexity is only possible when every step is pinned down overwhelmingly. Today, one of the chief pieces of advice I give to aspiring young rationalists is “Do not attempt long chains of reasoning or complicated plans.”""",
    """Once upon a time, in my reckless youth, when I knew not the Way of Bayes, I gave a Mysterious Answer to a mysterious-seeming question. Many failures occurred in sequence, but one mistake stands out as most critical: My younger self did not realize that solving a mystery should make it feel less confusing. I was trying to explain a Mysterious Phenomenon—which to me meant providing a cause for it, fitting it into an integrated model of reality. Why should this make the phenomenon less Mysterious, when that is its nature? I was trying to explain the Mysterious Phenomenon, not render it (by some impossible alchemy) into a mundane phenomenon, a phenomenon that wouldn’t even call out for an unusual explanation in the first place.

As a Traditional Rationalist, I knew the historical tales of astrologers and astronomy, of alchemists and chemistry, of vitalists and biology. But the Mysterious Phenomenon was not like this. It was something new, something stranger, something more difficult, something that ordinary science had failed to explain for centuries—

—as if stars and matter and life had not been mysteries for hundreds of years and thousands of years, from the dawn of human thought right up until science finally solved them—

We learn about astronomy and chemistry and biology in school, and it seems to us that these matters have always been the proper realm of science, that they have never been mysterious. When science dares to challenge a new Great Puzzle, the children of that generation are skeptical, for they have never seen science explain something that feels mysterious to them. Science is only good for explaining scientific subjects, like stars and matter and life.""",
    """There is a habit of thought which I call the logical fallacy of generalization from fictional evidence. Journalists who, for example, talk about the Terminator movies in a report on AI, do not usually treat Terminator as a prophecy or fixed truth. But the movie is recalled—is available—as if it were an illustrative historical case. As if the journalist had seen it happen on some other planet, so that it might well happen here.

There is an inverse error to generalizing from fictional evidence: failing to be sufficiently moved by historical evidence. The trouble with generalizing from fictional evidence is that it is fiction—it never actually happened. It’s not drawn from the same distribution as this, our real universe; fiction differs from reality in systematic ways. But history has happened, and should be available.

In our ancestral environment, there were no movies; what you saw with your own eyes was true. Is it any wonder that fictions we see in lifelike moving pictures have too great an impact on us? Conversely, things that really happened, we encounter as ink on paper; they happened, but we never saw them happen. We don’t remember them happening to us.""",
    """As our tribe wanders through the grasslands, searching for fruit trees and prey, it happens every now and then that water pours down from the sky.

“Why does water sometimes fall from the sky?” I ask the bearded wise man of our tribe.

He thinks for a moment, this question having never occurred to him before, and then says, “From time to time, the sky spirits battle, and when they do, their blood drips from the sky.”

“Where do the sky spirits come from?” I ask.

His voice drops to a whisper. “From the before time. From the long long ago.”

When it rains, and you don’t know why, you have several options. First, you could simply not ask why—not follow up on the question, or never think of the question in the first place. This is the Ignore command, which the bearded wise man originally selected. Second, you could try to devise some sort of explanation, the Explain command, as the bearded man did in response to your first question. Third, you could enjoy the sensation of mysteriousness—the Worship command.""",
    """Imagine that I, in full view of live television cameras, raised my hands and chanted abracadabra and caused a brilliant light to be born, flaring in empty space beyond my outstretched hands. Imagine that I committed this act of blatant, unmistakeable sorcery under the full supervision of James Randi and all skeptical armies. Most people, I think, would be fairly curious as to what was going on.

But now suppose instead that I don’t go on television. I do not wish to share the power, nor the truth behind it. I want to keep my sorcery secret. And yet I also want to cast my spells whenever and wherever I please. I want to cast my brilliant flare of light so that I can read a book on the train—without anyone becoming curious. Is there a spell that stops curiosity?

Yes indeed! Whenever anyone asks “How did you do that?” I just say “Science!”

It’s not a real explanation, so much as a curiosity-stopper. It doesn’t tell you whether the light will brighten or fade, change color in hue or saturation, and it certainly doesn’t tell you how to make a similar light yourself. You don’t actually know anything more than you knew before I said the magic word. But you turn away, satisfied that nothing unusual is going on.""",
]

with open("data/originals.json", "w") as f:
    json.dump(articles, f)
