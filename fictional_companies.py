import pandas as pd

# Replace this import with your actual LLMOpenAI class import
from src.utils.open_ai_utils import LLMOpenAI

class DatasetCreator:
    def __init__(self, llm):
        self.llm = llm  # Instance of LLMOpenAI class

    def generate_data(self, companies, stages):
        """
        Generate data for all companies across all stages.
        :param companies: List of company names.
        :param stages: List of stage prompts.
        :return: Dictionary of company data by stage.
        """
        data = {company: {} for company in companies}

        for stage_idx, stage_prompt in enumerate(stages):
            prompts = []
            for company in companies:
                context = "".join([f"\n\nStage {i+1}: {data[company].get(i+1, '')}" for i in range(stage_idx)])
                prompt = context + f"\n\n{stage_prompt.format(company=company)}"
                prompts.append([{"role": "user", "content": prompt}])
                # API call to generate data
            response = self.llm.generate(prompts, sampling_params={"temperature": 0.7, "max_tokens": 16000})

            for company, response in zip(companies, response):

                # Store the response for the current stage
                data[company][stage_idx + 1] = response

        return data

    def save_to_csv(self, data, filename):
        """
        Save generated data to a CSV file.
        :param data: Dictionary of company data by stage.
        :param filename: Name of the CSV file.
        """
        fieldnames = ['company', 'profile', 'catalogue', 'financials']
        data_df = pd.DataFrame(data).T.reset_index()
        data_df.columns = fieldnames
        data_df.to_csv(filename, index=False)
          
if __name__ == "__main__":
    # company names generated previously
    fictional_companies = [
    "MegaMarvelousMilkshakes Unlimited: A milkshake franchise started by two college dropouts who wanted to 'reinvent the blender.'",
    "QuantumQuirkySprouts Labs: An agricultural biotech startup founded by a retired quantum physicist and her vegan daughter.",
    "CozyCloudCottagesPlus Inc.: A prefab tiny-home manufacturer run by a husband-and-wife team who met at an architecture convention.",
    "All-DayAlpacaAwesome Apparel LLC: An eco-friendly clothing brand specializing in alpaca wool, owned by a Bolivian-American entrepreneur.",
    "ZapZapZoom Electric Scooters Co.: A ride-sharing company for electric scooters, founded by a former circus performer who dreamed of a quieter commute.",
    "TurboTerrificTacos LLC: A fast-casual Mexican restaurant chain created by a second-generation chef aiming to bring 'speed to spice.'",
    "DoodleDreamDelight Publishing Ltd.: A boutique children’s book publisher started by a former kindergarten teacher who wanted to focus on stories about diversity.",
    "GigaGreenGadgetry Inc.: A tech company specializing in solar-powered electronics, co-founded by two siblings with a love for camping.",
    "NeptuneNomadsNext Tours LLC: A high-end scuba diving and eco-tourism company run by a marine biologist and her former Navy SEAL husband.",
    "PolymerPalsPower Industries: A toy company known for customizable action figures, founded by a chemistry teacher who invented a new type of plastic.",
    "EpicExcellentEmpanadas LLC: A gourmet food truck business that began as a family recipe passed down through four generations.",
    "NextGenNectarNoble Brewing Co.: A craft brewery that uses AI to design unique beer recipes, founded by a former software engineer.",
    "SuperNovaSuperbSnacks Inc.: A healthy snack company targeting astronauts and space enthusiasts, started by a former NASA intern.",
    "ZebraZenZesty Yoga Studios: A yoga chain with zebra-themed decor, founded by an eccentric wildlife photographer turned wellness coach.",
    "ThunderTrucksTitanic Hauling LLC: A trucking company specializing in oversized cargo, run by a family with a three-generation history in logistics.",
    "ByteBloomBrilliant Innovations: A cybersecurity firm with a focus on startups, founded by a reclusive coder known only as 'Bloom.'",
    "OasisOrganicsOutstanding Skincare Ltd.: A luxury skincare line that uses desert botanicals, started by a hydrologist and a dermatologist in Arizona.",
    "RoboRustRadical R&D Labs: A robotics company designing post-apocalyptic-themed robots, founded by two gamers turned engineers.",
    "PixelPeachPrime Animation Studios: An indie animation house known for quirky 8-bit style films, started by a graphic designer and her gamer husband.",
    "CivicSynergySuper Solutions Inc.: A consulting firm focused on urban planning and civic technology, run by a former city planner and a sociologist.",
    "ExtraExcellentChairs Inc.: A chair manufacturer known for producing the comfiest chairs in the industry.",
    "SuperAmazingGreatSoap Inc.: A soap manufacturer founded by Pakistani immigrants with a mission to create high-quality and affordable soap.",
    "FormerTechGiant LLC: A tech giant in decline, remembered for its glory days in innovation and cutting-edge technology."
    ]

    # Prompts for each stage
    stage_prompts = [
"""I want to create a dataset of fictional companies. Each company should resemble real companies in general, but not mimic any specific company directly. 

Let's start with "{company}" Provide a comprehensive company profile, similar to what you might find in a detailed Wikipedia article about a major corporation.

The profile should include:

Company history and background (e.g., founders, founding year, and early milestones). 
Key financial highlights (e.g., market capitalization, revenue, major funding rounds, or IPO details). 
A detailed description of its products, services, or business model, including major innovations and failures. 
Current corporate structure, key executives, and employee headcount. 
Offices, headquarters, and geographical reach. 
Recent controversies, challenges, or notable news (e.g., lawsuits, scandals, failed products, or market competition). 
Details about its industry standing, competitors, and long-term strategic goals. 
Write the response as detailed, paragraphed, and well-structured text. Use an informative, neutral tone similar to that of a Wikipedia article. Incorporate small, specific details to make the profile feel authentic and nuanced.""",

"""Based on the company profile provided, create a detailed product catalog for "{company}" Ensure it is as rich and detailed as a real-world product catalog. 

The catalog should include:

At least 20 products, spanning multiple categories that align with the company's business model. 
Full product descriptions, written with a mix of marketing language and technical details. 
Comprehensive specifications for each product, including features, technical specs, and any unique selling points. 
Pricing details (e.g., MSRP, subscription fees, or tiers for different customer segments). 
Any notable reviews, industry recognition, or customer use cases for certain products. 
Notes on availability (e.g., discontinued products, regional exclusivity, or supply issues). 
Write this in unstructured, free-form text. Think of how a large company might present its offerings in an online store, brochure, or detailed catalog. Ensure the tone is professional but engaging.""",

"""Recall the company profile and product catalog for "{company}" Create a full financial report of its performance for the past quarter. The report should be consistent with the previous information provided and sound realistic from an accounting and corporate reporting perspective. 

The financial report should include:

Revenue, gross profit, and net income figures, broken down by business segments (e.g., product lines, geographical regions, or divisions). 
Key financial ratios (e.g., profit margins, debt-to-equity ratio, etc.). 
Year-over-year (YoY) or quarter-over-quarter (QoQ) comparisons, highlighting trends. 
Major revenue drivers and cost challenges, including any standout successes or failures. 
Commentary on the company's cash flow, liquidity, and capital expenditures. 
Updates on stock performance, shareholder returns, or dividend announcements (if applicable). 
Notable financial risks or opportunities (e.g., lawsuits, declining market share, or upcoming product launches). 
Write this in detailed, unstructured free-form text, as if prepared by the finance department of a major corporation for internal or investor reporting. Include realistic terminology and phrasing to create an authentic tone."""
    ]

    # Initialize the LLM class instance
    llm = LLMOpenAI(model='gpt-4o')  # Replace with your actual LLMOpenAI class initialization

    # Generate dataset
    creator = DatasetCreator(llm)
    company_data = creator.generate_data(fictional_companies, stage_prompts)

    # Save to CSV
    creator.save_to_csv(company_data, "fictional_companies_dataset.csv")
    print("Dataset saved to fictional_companies_dataset.csv.")
