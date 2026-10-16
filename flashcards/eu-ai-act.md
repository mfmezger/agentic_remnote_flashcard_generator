# Flashcards: EU AI Act

## Basics and Scope

- What is the official designation of the EU AI Act? == Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence.
- What overall regulatory philosophy does the AI Act follow? == A risk-based approach: obligations scale with risk, from prohibited practices through high-risk and transparency obligations to minimal-risk systems with no specific obligations.
- Does the AI Act apply to a provider established outside the EU? == Yes, if it places AI systems or GPAI models on the EU market or puts them into service in the EU, regardless of where it is established.
- When does the AI Act apply to non-EU providers and deployers even if the system is not placed on the EU market? == When the output produced by the AI system is used in the Union.
- Name the main exclusions from the AI Act's scope. == Military, defence and national security uses; AI developed solely for scientific research and development; purely personal non-professional use; and research and testing before placing on the market (not real-world testing).
- When are AI systems released under free and open-source licences still covered by the AI Act? == When they are placed on the market as high-risk, fall under prohibited practices, or are subject to the Article 50 transparency obligations.
- How does the AI Act relate to the GDPR? == It applies without prejudice to EU data protection law; GDPR obligations continue to apply alongside it whenever personal data is processed.

## Definition of an AI System

- What are the key elements of the AI Act's definition of an AI system? == A machine-based system with varying levels of autonomy, possibly adaptive after deployment, that infers from input how to generate outputs such as predictions, content, recommendations or decisions influencing physical or virtual environments.
- Which capability in the AI system definition mainly distinguishes AI from simple traditional software? == The capability to infer how to generate outputs from the input received, rather than only executing rules defined solely by humans.

## Roles in the Value Chain

- Who is a provider under the AI Act? == A person or body that develops an AI system or GPAI model, or has it developed, and places it on the market or puts it into service under its own name or trademark, paid or free.
- Who is a deployer under the AI Act? == A natural or legal person, public authority or body using an AI system under its authority, except for personal non-professional activity.
- Who is an importer under the AI Act? == A person established in the Union who places on the market an AI system bearing the name or trademark of a person established in a third country.
- Who is a distributor under the AI Act? == A person in the supply chain, other than the provider or importer, who makes an AI system available on the Union market.
- What is an authorised representative under the AI Act? == A person established in the Union holding a written mandate from a non-EU provider to perform the provider's obligations and procedures on its behalf.
- What is a downstream provider? == A provider of an AI system, including a GPAI system, that integrates an AI model, whether its own or one supplied by another entity.
- When does a deployer become the provider of a high-risk system by branding? == When it puts its own name or trademark on a high-risk AI system already placed on the market or put into service.
- When does a substantial modification turn a deployer into a provider? == When it makes a substantial modification to a high-risk AI system such that the system remains high-risk.
- How can repurposing a GPAI-based chatbot make a company a high-risk provider? == Modifying the intended purpose of a non-high-risk AI system, including a GPAI system, so that it becomes high-risk makes the modifier the provider.

## Prohibited Practices

- Which article of the AI Act lists prohibited AI practices? == Article 5.
- What is prohibited regarding manipulative AI techniques? == Subliminal, purposefully manipulative or deceptive techniques that materially distort behaviour by impairing informed decision-making, causing or likely to cause significant harm.
- What is prohibited regarding exploitation of vulnerabilities? == AI exploiting vulnerabilities due to age, disability or a specific social or economic situation to materially distort behaviour in a way causing or likely to cause significant harm.
- What social scoring is prohibited by the AI Act? == Evaluating or classifying people based on social behaviour or personal traits, leading to detrimental treatment in unrelated contexts or treatment unjustified or disproportionate to their behaviour, by public or private actors.
- What is prohibited regarding predictive policing? == Assessing the risk of a person committing a criminal offence based solely on profiling or personality traits, unless supporting a human assessment based on objective, verifiable facts linked to criminal activity.
- What is prohibited regarding facial image scraping? == Creating or expanding facial recognition databases through untargeted scraping of facial images from the internet or CCTV footage.
- Where is emotion recognition prohibited, and what is the exception? == In the workplace and education institutions, except where used for medical or safety reasons.
- What biometric categorisation is prohibited? == Categorising people from biometric data to deduce race, political opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation.
- What use of real-time remote biometric identification is prohibited by default? == Its use in publicly accessible spaces for the purposes of law enforcement.
- Name the three exceptions allowing real-time remote biometric identification for law enforcement. == Targeted search for victims of abduction, trafficking or sexual exploitation and missing persons; preventing a specific imminent threat to life or a terrorist attack; locating or identifying suspects of listed serious crimes.

## High-Risk Classification

- What are the two routes by which an AI system becomes high-risk? == Annex I: safety component of, or itself, a product under listed EU product legislation requiring third-party conformity assessment; or Annex III: listed sensitive use cases.
- Which areas are listed in Annex III? == Biometrics, critical infrastructure, education, employment and worker management, access to essential services, law enforcement, migration and border control, and administration of justice and democratic processes.
- Give examples of high-risk employment/HR use cases under Annex III. == AI for recruitment or selection such as targeted job ads, filtering applications and evaluating candidates, and AI for promotion, termination, task allocation or monitoring and evaluating workers.
- Which credit-related use case is high-risk under Annex III, and what is excluded? == Evaluating creditworthiness or establishing credit scores of natural persons; AI used to detect financial fraud is excluded.
- What does the Article 6(3) derogation allow? == An Annex III system is not high-risk if it poses no significant risk of harm, e.g. performing a narrow procedural or preparatory task, or improving the result of a previously completed human activity.
- When can the Article 6(3) derogation never apply to an Annex III system? == When the AI system performs profiling of natural persons; it is then always high-risk.
- What must a provider do when relying on the Article 6(3) derogation? == Document its assessment before placing the system on the market and register the system in the EU database; authorities may request the documentation.

## High-Risk Provider Obligations

- What must the risk management system for a high-risk AI system be? == A continuous, iterative process planned and run throughout the entire lifecycle to identify, analyse, evaluate and mitigate risks to health, safety and fundamental rights.
- What does data governance require for high-risk AI systems? == Training, validation and testing data must be relevant, sufficiently representative, as error-free and complete as possible, and examined for possible biases.
- What is the purpose of technical documentation for high-risk AI systems? == To demonstrate compliance and give authorities and notified bodies the information to assess it; it must be drawn up before market placement and kept up to date.
- What logging capability must high-risk AI systems have? == They must technically allow automatic recording of events (logs) over their lifetime to ensure traceability.
- What transparency must providers give deployers of high-risk systems? == Instructions for use with concise, clear information on provider identity, intended purpose, performance, accuracy, limitations, risks, human oversight measures and log handling.
- What does human oversight design for high-risk systems aim to enable? == Natural persons can understand capabilities and limits, avoid automation bias, correctly interpret outputs, and decide to disregard, override or stop the system.
- Which technical quality properties must high-risk AI systems achieve? == An appropriate level of accuracy, robustness and cybersecurity throughout their lifecycle, including resilience against attacks like data poisoning and adversarial examples.
- What is the quality management system obligation? == Providers must maintain a documented QMS covering compliance strategy, design and development, testing, data management, risk management, post-market monitoring and incident reporting.
- What must happen before a high-risk AI system is placed on the market? == It must undergo the applicable conformity assessment, after which the provider draws up an EU declaration of conformity and affixes the CE marking.
- Which conformity assessment procedure applies to most Annex III high-risk systems? == Internal control by the provider (self-assessment); notified body involvement mainly concerns certain biometric systems and Annex I products.
- Which high-risk systems must be registered in the EU database before market placement? == Annex III high-risk systems (registered by the provider; critical-infrastructure systems are registered at national level instead), with public-authority deployers also registering their use.
- What is post-market monitoring? == A provider-run system to actively collect and analyse data on the performance of high-risk systems over their lifetime to evaluate continued compliance.
- What is the general deadline for providers to report a serious incident? == Immediately after establishing a causal link or its reasonable likelihood, and no later than 15 days after becoming aware; shorter deadlines apply to the most severe cases.

## High-Risk Deployer Obligations

- What is a deployer's basic obligation when using a high-risk AI system? == Take appropriate technical and organisational measures to use it in accordance with the provider's instructions for use.
- Whom must deployers assign to human oversight? == Natural persons with the necessary competence, training and authority, plus the necessary support.
- What must deployers ensure about input data? == To the extent they control it, input data must be relevant and sufficiently representative for the system's intended purpose.
- How long must deployers retain logs of high-risk AI systems? == At least six months, to the extent logs are under their control, unless other law provides otherwise.
- What must employers do before using a high-risk AI system in the workplace? == Inform workers' representatives and the affected workers that they will be subject to the system.
- Which deployers must perform a fundamental rights impact assessment (FRIA)? == Public bodies and private entities providing public services, plus deployers using AI for creditworthiness or credit scoring and for life and health insurance risk pricing.
- What right do affected persons have regarding decisions based on Annex III high-risk systems? == A right to obtain clear and meaningful explanations of the AI system's role in a decision producing legal or similarly significant effects on them.

## Transparency Obligations

- What must providers of chatbots and other AI interacting with people ensure? == People are informed that they are interacting with an AI system, unless obvious to a reasonably well-informed person from the context.
- What must providers of generative AI producing synthetic audio, image, video or text ensure? == Outputs are marked in a machine-readable format and detectable as artificially generated or manipulated, as far as technically feasible.
- What must deployers of deepfakes do? == Disclose that the image, audio or video content has been artificially generated or manipulated, with lighter obligations for evidently artistic, satirical or fictional works.
- When must deployers disclose AI-generated text? == When it is published to inform the public on matters of public interest, unless it underwent human review or editorial control with editorial responsibility.
- What must deployers of emotion recognition or biometric categorisation systems do? == Inform the natural persons exposed to the system about its operation and process personal data in line with data protection law.

## General-Purpose AI Models

- What is a general-purpose AI (GPAI) model? == A model, often trained on large data with self-supervision at scale, showing significant generality, competently performing a wide range of distinct tasks and integrable into various downstream systems.
- What are the four core obligations of all GPAI model providers? == Keep technical documentation, provide information to downstream providers, put in place a copyright compliance policy, and publish a sufficiently detailed training content summary.
- What must the GPAI copyright policy address in particular? == Identifying and respecting rights reservations (opt-outs) from text and data mining under EU copyright law, including via state-of-the-art technologies.
- When is a GPAI model presumed to have systemic risk? == When the cumulative compute used for its training exceeds 10^25 floating-point operations (FLOPs).
- Can a GPAI model be classified as systemic-risk without crossing the compute threshold? == Yes, the Commission can designate it based on high-impact capabilities, on its own initiative or after a qualified alert from the scientific panel.
- What additional obligations apply to GPAI models with systemic risk? == State-of-the-art model evaluation including adversarial testing, assessing and mitigating systemic risks, tracking and reporting serious incidents, and ensuring adequate cybersecurity for the model and its infrastructure.
- What open-source exemption exists for GPAI providers? == Models under a free and open-source licence with publicly available weights, architecture and usage information are exempt from technical documentation and downstream information duties.
- What are the limits of the GPAI open-source exemption? == It does not cover the copyright policy or training content summary, and does not apply at all to GPAI models with systemic risk.
- What is the GPAI Code of Practice? == A voluntary code published in July 2025 with transparency, copyright, and safety and security chapters that providers can sign to demonstrate compliance with GPAI obligations.
- By when must GPAI models placed on the market before 2 August 2025 comply? == By 2 August 2027.
- Who supervises and enforces GPAI model obligations? == The European Commission, which entrusts these tasks to its AI Office; the Commission can also impose fines on GPAI providers.

## AI Literacy

- What does the Article 4 AI literacy obligation require? == Providers and deployers must take measures to ensure, to their best extent, a sufficient level of AI literacy of staff and others operating or using AI systems on their behalf.
- Does the AI literacy obligation apply only to high-risk AI systems? == No, it applies to providers and deployers of all AI systems.

## Governance

- What is the AI Office? == A body within the European Commission that develops Union expertise on AI, supervises GPAI models and supports consistent application of the AI Act.
- What is the European Artificial Intelligence Board? == A body of Member State representatives advising and assisting the Commission and Member States to ensure consistent and effective application of the Act.
- What national authorities must each Member State designate? == At least one notifying authority and at least one market surveillance authority as national competent authorities.

## Penalties

- What is the maximum fine for violating the prohibited practices? == Up to EUR 35 million or 7% of total worldwide annual turnover for the preceding financial year, whichever is higher.
- What is the maximum fine for non-compliance with most other operator obligations, including transparency? == Up to EUR 15 million or 3% of total worldwide annual turnover, whichever is higher.
- What is the maximum fine for supplying incorrect, incomplete or misleading information to authorities or notified bodies? == Up to EUR 7.5 million or 1% of total worldwide annual turnover, whichever is higher.
- How are fine caps applied differently to SMEs and start-ups? == The lower of the fixed amount and the turnover percentage applies, instead of the higher.

## Timeline

- When did the AI Act enter into force? == On 1 August 2024, twenty days after publication in the Official Journal.
- What started applying on 2 February 2025? == The general provisions including the AI literacy obligation, and the prohibited practices.
- What started applying on 2 August 2025? == GPAI model obligations, governance provisions, notified body rules and most penalty provisions.
- What applies from 2 August 2026 under the original regulation? == Most remaining obligations, including Annex III high-risk requirements and Article 50 transparency obligations.
- What applies from 2 August 2027 under the original regulation? == Obligations for high-risk systems under the Annex I product-safety route (Article 6(1)).
- Why must the high-risk application dates be double-checked? == In November 2025 the Commission proposed a Digital Omnibus that would postpone some high-risk obligations; verify the current legal status before relying on the original dates.

## Regulatory Sandboxes

- What is an AI regulatory sandbox? == A controlled framework set up by a competent authority where providers can develop, train, validate and test innovative AI systems under regulatory supervision for a limited time.
- What must each Member State ensure regarding sandboxes? == At least one AI regulatory sandbox at national level operational by 2 August 2026, possibly jointly with other Member States.
- Which organisations get priority access to AI regulatory sandboxes? == SMEs, including start-ups, which receive priority access generally free of charge.
