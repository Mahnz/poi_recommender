from lib.poi_logger import POILog

tag = "Input Validator"


def validate_input(prompt, max_length=-1, type=str):
    while True:
        POILog.i(tag, "", suffix="")
        user_input = input(prompt)

        valid = True

        if max_length > 0:
            valid = valid and len(user_input) <= max_length
        if type == int:
            valid = valid and user_input.isnumeric()

        if valid:
            return user_input
        else:
            if type == str:
                POILog.w(tag, f"Input must be {max_length} characters or fewer. Please try again.")
            elif type == int:
                POILog.w(tag, f"Input must be numeric. Please try again.")
            else:
                POILog.e(tag, f"Input type not recognized.")
                return None


def main():
    user_description = validate_input("Please enter the description of the desired venue (max 50 characters): ", 50)
    print(int(user_description))


if __name__ == "__main__":
    main()
