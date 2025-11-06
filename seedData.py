import pandas as pd
import numpy as np

# Generate mock training data
n_samples = 10
n_questions = 20

data = {f'Q{i+1}': np.random.randint(1, 6, size=n_samples) for i in range(n_questions)}
data['label'] = np.random.randint(1, 12, size=n_samples)  # Tech field IDs 1-11

df = pd.DataFrame(data)

import ace_tools as tools; tools.display_dataframe_to_user(name="Mock Training Dataset", dataframe=df)
