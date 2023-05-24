Student Name: Wong Man Cong

Project Name: Fast Food Restaurant

What I implemented:
I implemented an AI behaviour tree whereby it mimics behaviours of customers, cashier, cook and a manager

Following is the rough idea of the behaviour of each AI

Customers:
1) Walk across store
2) Decide if want to eat
-> Change colour 
-> If customer will be eating, go into store (3)
-> Else continue walking
3) Get in line and wait for turn
4) Order food
5) Decide to eat in store or have it to go (6)
6) Finish food and leave the restaurant

Cashier:
-> Might screw up, if screw up more than 2 times will be fired
-> Take order
-> When food is ready, go and collect it and serve to customer
1) Wait for customer
2) Get order from customer
3) Wait for chief to finish cooking
4) Once chef finish cooking, collect food
5) Serve to customer

Chef:
1) Wait for order
2) Cook dish (Spinning about the Y axis)
3) Chance to mess up the dish (turn red and rotate about the Z axis quickly)
4) Serve dish

Manager:
1) Take orders
2) Serve food to customers
3) Check in on kitchen
4) Check in on cashier

Directions (if needed):

What I liked about the project and framework:
- There is this class in the framework called blackboard, i think it's a very interesting concept
- The naming convention for the control flow, decorators and leaf nodes

What I disliked about the project and framework:
- The use of the keyword "auto". It makes understanding the project a little more difficult
- Not being able to load more than one model
- Closing the console leads to breaking of program
- Using the Behaviour Tree Editor, whenever I were to have an interaction with it, the nodes keep uncollasping making it very annoying

Any difficulties I experienced while doing the project:
Had some difficulties thinking of what I wanna do initially, however after
I got the idea of making the AI based on a restaurant, everything works as
intended

Hours spent: 
10-15hrs

New selector node (name):

New decorator nodes (names):
D_AlwaysSucceed
D_OnlyRunOnce
D_CheckIfDecided
D_Inverter
D_CheckForCustomer
D_Timer
D_ChanceOfScrewUp
D_CounterToGetFired
D_RunTillSucceed
D_ChanceToMessUpDish

10 total nodes (names):
L_RandomPointOutsideStore
L_WalkAcrossStore
L_ChangeAgentColour
L_WalkIntoStore
L_GetInLine
L_MoveTowardsCashier
L_ItsMyTurn
L_OrderMessedUp
L_ResetCustomerVariables
L_EmptyTable
L_WalkToTable
L_EatAtTable
L_HaveItToGo
L_LeaveStore
L_CheckOnCashier
L_CheckOnChef
L_WaitForCustomer
L_TakeOrderFromCustomer
L_WaitForChef
L_GoToKitchen
L_CollectFood
L_ServeCustomer
L_GetsFired
L_RespawnCashier
L_WaitForOrder
L_CookDish
L_AngryCook
L_ServeDish
L_AngryCustomer
L_SpawnCustomers
L_Idle


4 Behavior trees (names):
Customer
Manager
Cook
Cashier
LeaveStoreAngrily
CustomerSpawner

Extra credit: