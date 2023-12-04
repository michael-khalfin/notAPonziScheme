import pandas as pd

# FINAL_PERIODS = 60
# INITIAL_PERIODS = 72
DATA_PATH = "market_return_cap.csv"


def read_data(path):
    return pd.read_csv(path)


def get_year(date):

    date_list = str(date).split('/')
    return int(date_list[2])


def generate_output(periods, data):

    names = []  # Stores the stock names in order
    returns = []  # Stores lists containing the returns for each period, for each stock
    caps = []  # Same, but for capitalizations

    for _ in range(periods):
        returns.append([])
        caps.append([])

    stock = None
    encounter = 0
    for index, row in data.iterrows():


        stock_1 = row.iloc[0]  # Read the row data
        date = row.iloc[1]
        cap = row.iloc[2]
        ret = row.iloc[3]

        # print(f"Stock: {stock_1}", f" Return: {ret}")

        if get_year(date) < 2017:  # Isolate year, and check if we care about this period
            pass
        elif stock_1 == 'LIN' or stock_1 == 'CAN' or stock_1 == 'META' or stock_1 == 'FB':  # Remove stocks with insufficient data
            pass
        else:
            if stock == stock_1:
                encounter += 1
                if encounter > periods-1:
                    continue
                returns[encounter].append(ret)
                caps[encounter].append(cap)
            else:
                # print(f"Stock: {stock} found {encounter} times")
                encounter = 0
                stock = stock_1
                names.append(stock)
                returns[encounter].append(ret)
                caps[encounter].append(cap)
    return names, returns, caps

def create_file(periods, path):
    data = read_data(path)
    period_data = generate_output(periods, data)


    names = period_data[0]
    returns = period_data[1]
    caps = period_data[2]

    # print(len(names))
    # print(len(caps))
    # print(len(returns))

    market_returns = pd.DataFrame({'names': names})
    capitalizations = pd.DataFrame({'names': names})

    for i in range(periods):
        # if i < start:
          #  continue
        # print(len(returns[i]))
        # print(len(caps[i]))
        market_returns[f"{i}"] = returns[i]
        capitalizations[f"{i+1}"] = caps[i]

    market_return_path = f"periodic_returns_{periods}.csv"
    capitalization_path = f"periodic_caps_{periods}.csv"

    market_returns.to_csv(market_return_path, index=False, encoding='utf-8')
    capitalizations.to_csv(capitalization_path, index=False, encoding='utf-8')


def generate_all_files(path):
    print("here")
    for i in range(60, 72):
        create_file(i, path)


generate_all_files(DATA_PATH)



