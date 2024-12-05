from content_based_pipeline import run_pipeline_cb
from collaborative_based_pipeline import run_pipeline_cf
from hybird_based_pipeline import run_pipeline_hb
from training.model_training.Data_Ingestion import file_path
# the RecommendationSystem class
class RecommendationSystem:
    def __init__(self):
        print("Recommendation System initialized.")

    # Methods corresponding to the imported functions
    def run_function1(self):
        return run_pipeline_cb(file_path)

    def run_function2(self):
        return run_pipeline_cf(file_path)

    def run_function3(self):
        return run_pipeline_hb(file_path)

    # User interface method
    def user_interface(self):
        while True:
            print("Select the function to execute:")
            print("1: Content based press 1")
            print("2: Collaborative filtering based press 2")
            print("3: Hybird based press 3")
            print("0: Exit")
            
            choice = input("Enter your choice: ")
            if choice == "1":
                print(self.run_function1())
            elif choice == "2":
                print(self.run_function2())
            elif choice == "3":
                print(self.run_function3())
            elif choice == "0":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

# Example of how to run the system
if __name__ == "__main__":
    system = RecommendationSystem()
    system.user_interface()
