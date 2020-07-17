#
#   hw1.py
#
#    Ethan Sorkin, Robin Park
#    February 2020
#    COMP 131: Artificial Intelligence
#


# Constants
BATTERY_MIN    = 30
BATTERY_MAX    = 100
SPOT_DURATION  = 20
DUSTY_DURATION = 35

class Roomba:

    def __init__(self, data):
        self.blackboard = data
        self.global_timer = 0
        self.spot_timer = 0
        self.dusty_timer = 0

    # Returns path to docking station
    def find_home(self):
        print("Find Home - RUNNING")
        print("Find Home - SUCCEEDED; Go " + self.blackboard["HOME_PATH"])
        return self.blackboard["HOME_PATH"]

    # Follow given path to docking station
    def go_home(self, path):
        print("Go Home - RUNNING")
        print("Go Home - SUCCEEDED")

    # Docking process to charge battery
    def dock(self):
        while self.blackboard["BATTERY_LEVEL"] < BATTERY_MAX:
            self.blackboard["BATTERY_LEVEL"] += 1
            print("Dock - RUNNING; Battery Level=" + \
                  str(self.blackboard["BATTERY_LEVEL"]) + "%")
        print("Dock - SUCCEEDED; Full Battery")

    # P1: Battery check / charge
    def P1(self):
        if self.blackboard["BATTERY_LEVEL"] < BATTERY_MIN:
            if self.blackboard["GENERAL"] and self.global_timer > 0:
                self.done_general()
                if self.blackboard["DUSTY_SPOT"]:
                    self.fail_dusty_spot()
            self.go_home(self.find_home())
            self.dock()

    # Spot cleaning: perform a 20s intensive cleaning in a specific area
    def clean_spot(self):
        print("Spot Clean - RUNNING")

    # Changes value of SPOT when spot clean is complete
    def done_spot(self):
        self.blackboard["SPOT"] = False
        print("Spot Clean - SUCCEEDED")

    # Changes value of DUSTY_SPOT when 35s spot clean is complete
    def done_dusty_spot(self):
        self.blackboard["DUSTY_SPOT"] = False
        print("Spot Clean (Dusty) - SUCCEEDED")

    # Changes value of DUSTY_SPOT when it fails
    def fail_dusty_spot(self):
        print("Spot Clean (Dusty) - FAILED")
        self.blackboard["DUSTY_SPOT"] = False

    # General cleaning: go around the room and vacuum dust until the battery
    # falls under 30%
    def clean(self):
        print("General Clean - RUNNING")

    # Changes value of GENERAL when general clean is complete
    def done_general(self):
        self.blackboard["GENERAL"] = False
        print("General Clean - SUCCEEDED")

    # P2: Cleaning process
    def P2(self):
        # Spot Clean
        if self.blackboard["SPOT"]:
            if self.spot_timer < SPOT_DURATION:
                self.clean_spot()
                self.spot_timer += 1
            else:
                self.done_spot()

        # General clean
        if self.blackboard["GENERAL"]:
            if self.blackboard["BATTERY_LEVEL"] >= BATTERY_MIN:
                if self.blackboard["DUSTY_SPOT"]:
                    if self.dusty_timer < DUSTY_DURATION:
                        self.clean_spot()
                        self.dusty_timer += 1
                    else:
                        self.done_dusty_spot()
                self.clean()
            else:
                print("General Cleaning - FAILED; Insufficient battery")
        elif self.blackboard["DUSTY_SPOT"]:
            self.fail_dusty_spot()


    # P3: Do Nothing
    def P3(self):
        print("Do Nothing")

    # Returns true if any task is incomplete
    def check_progress(self):
        return self.blackboard["SPOT"] or self.blackboard["GENERAL"] \
               or self.blackboard["DUSTY_SPOT"]

    def run(self):
        while self.check_progress():
            self.P1()
            self.P2()
            self.P3()
            self.global_timer += 1
            self.blackboard["BATTERY_LEVEL"] -= 1

        print("All Tasks Completed")
        print("Shutting down")

def main():
    # Battery Level Tests:
    test1 = Roomba({"BATTERY_LEVEL": 100, "SPOT": True, "GENERAL": True,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test1.run()
    # Expected result: All tasks will succeed.
    # Order of completion: Spot, General, Dusty Spot

    test2 = Roomba({"BATTERY_LEVEL": 64, "SPOT": True, "GENERAL": True,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test2.run()
    # Expected result: Dusty spot clean will fail, because the battery
    # will fall below 30% before it has succeeded.
    # Order of completion: Spot, General, Dusty Spot (fail)

    test3 = Roomba({"BATTERY_LEVEL": 35, "SPOT": True, "GENERAL": True,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test3.run()
    # Expected result: Dusty spot clean will fail because of insufficient
    # battery, and general clean will terminate before spot clean.
    # Order of completion: General, Dusty Spot (fail), Spot

    test4 = Roomba({"BATTERY_LEVEL": 20, "SPOT": True, "GENERAL": True,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test4.run()
    # Expected result: Battery will immediately charge to 100%, then carry
    # out all tasks the same way as test1
    # Order of completion: Spot, General, Dusty Spot


    # Blackboard Value Tests:
    test5 = Roomba({"BATTERY_LEVEL": 100, "SPOT": True, "GENERAL": True,
                    "DUSTY_SPOT": False, "HOME_PATH": "5 feet North"})
    test5.run()
    # Expected result: Spot and General succeed, Dusty Spot Clean is not executed
    # Order of completion: Spot, General

    test6 = Roomba({"BATTERY_LEVEL": 100, "SPOT": True, "GENERAL": False,
                    "DUSTY_SPOT": False, "HOME_PATH": "5 feet North"})
    test6.run()
    # Expected result: Spot clean succeeds, other tasks are not executed
    # Order of completion: Spot

    test7 = Roomba({"BATTERY_LEVEL": 100, "SPOT": True, "GENERAL": False,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test7.run()
    # Expected result: Spot clean succeeds, Dusty Spot clean fails on first
    # cycle because General is not executed
    # Order of completion: Dusty Spot (fail), Spot

    test8 = Roomba({"BATTERY_LEVEL": 100, "SPOT": False, "GENERAL": True,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test8.run()
    # Expected result: General and Dusty Spot succeed, Spot is never executed
    # Order of completion: General, Dusty Spot

    test9 = Roomba({"BATTERY_LEVEL": 100, "SPOT": False, "GENERAL": False,
                    "DUSTY_SPOT": True, "HOME_PATH": "5 feet North"})
    test9.run()
    # Expected result: Dusty Spot fails on first cycle, other tasks are not executed
    # Order of completion: Dusty Spot (fail)

    test10 = Roomba({"BATTERY_LEVEL": 100, "SPOT": False, "GENERAL": False,
                    "DUSTY_SPOT": False, "HOME_PATH": "5 feet North"})
    test10.run()
    # Expected result: Nothing executes, so it shuts down immediately
    # Order of completion: N/A

    test11 = Roomba({"BATTERY_LEVEL": 100, "SPOT": False, "GENERAL": True,
                    "DUSTY_SPOT": False, "HOME_PATH": "5 feet North"})
    test11.run()
    # Expected result: General succeeds, other tasks are not executed
    # Order of completion: General




if __name__ == "__main__":
    main(

)

