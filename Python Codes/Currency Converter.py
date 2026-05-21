def get_exchange_rates():
    """
    Return exchange rates from 1 NGN to target currencies.
    (Rates are example values; replace with live numbers as needed.)
    """
    return {
        "USD": 1 / 1180.50,  # approx 0.000847 USD per NGN
        "EUR": 1 / 1279.30,  # approx 0.000782 EUR per NGN
        "GBP": 1 / 1508.60,  # approx 0.000663 GBP per NGN
    }


def convert_currency(amount_ngn, rates):
    """
    Convert amount_ngn to each target currency using rates dict.
    Returns a dict with converted values.
    """
    converted = {}
    for currency, rate in rates.items():
        converted[currency] = amount_ngn * rate
    return converted


def format_output(amount_ngn, converted):
    print("\n=====Currency Conversion Result=====")
    print(f"Input (NGN): ₦{amount_ngn:,.2f}")
    print("------------------------------------")
    for currency, value in converted.items():
        symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
        print(f"{currency:>3}: {symbol}{value:,.2f}")
    print("====================================\n")


def main():
    rates = get_exchange_rates()
    while True:
        user_input = input("Enter amount in NGN (or type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            amount_ngn = float(user_input)
            if amount_ngn < 0:
                raise ValueError("Amount must be non-negative.")
        except ValueError:
            print("Invalid input. Enter a valid number (e.g., 1000) or 'exit'.")
            continue

        converted = convert_currency(amount_ngn, rates)
        format_output(amount_ngn, converted)


if __name__ == "__main__":
    main()