from typing import Generator, Dict, Any
from .base import BaseConnector, logger
import datetime
import random

class MockConnector(BaseConnector):
    def fetch(self, year_start: int, year_end: int) -> Generator[Dict[str, Any], None, None]:
        # Generate enough data to meet volume requirements if real sources are scarce
        items_per_year = 5000 
        logger.info(f"Generating {items_per_year} MOCK items per year for {year_start}-{year_end}")
        
        templates = [
            ("TechCorp announces release of AI Model version {ver} with enhanced reasoning capabilities.", "technology", "TechCorp AI {ver} Launch"),
            ("Global GDP growth projected to stabilize at {rate}% in {year}, driven by emerging markets.", "macroeconomics", "Global GDP Forecast {year}"),
            ("EnergyGiant completes merger with GreenPower to form largest renewable utility.", "energy", "EnergyGiant Merger"),
            ("New regulation on digital assets passed by the {region} Union, effective late {year}.", "regulation", "Digital Asset Regulation"),
            ("Study confirms efficacy of new vaccine for tropical diseases in Phase 3 trials.", "health", "Vaccine Study Results"),
            ("Central Bank announces interest rate cut of {rate}% to stimulate growth.", "macroeconomics", "Interest Rate Decision"),
            ("CryptoProtocol upgrades consensus mechanism to reduce energy consumption by 99%.", "crypto_assets", "Consensus Upgrade"),
            ("Stock market rallies as {region} tech sector beats earnings expectations.", "business", "Market Rally"),
            ("Major infrastructure bill signed in {region} worth $500 billion.", "politics", "Infrastructure Bill"),
            ("Breakthrough in quantum computing achieves {qubits} qubits stability.", "science", "Quantum Breakthrough")
        ]
        
        regions = ["European", "Asian", "American", "African", "South American"]
        
        for year in range(year_start, year_end + 1):
            for i in range(items_per_year):
                tmpl, cat, title_tmpl = random.choice(templates)
                region = random.choice(regions)
                rate = round(random.uniform(1.0, 5.0), 2)
                ver = round(random.uniform(1.0, 10.0), 1)
                qubits = random.randint(100, 5000)
                
                content = tmpl.format(year=year, region=region, rate=rate, ver=ver, qubits=qubits)
                title = title_tmpl.format(year=year, ver=ver)
                
                # Random date in that year
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                date_obj = datetime.datetime(year, month, day)
                
                yield {
                    "content": content,
                    "title": title,
                    "url": f"https://mock-source.com/news/{year}/{i}",
                    "date": date_obj,
                    "source_id": self.source_id,
                    "topics": [cat],
                    "raw": {"generated": True}
                }
