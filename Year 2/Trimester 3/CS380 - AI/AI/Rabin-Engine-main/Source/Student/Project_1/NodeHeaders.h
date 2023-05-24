#pragma once

// Include all node headers in this file

// Example Control Flow Nodes
#include "ControlFlow/C_ParallelSequencer.h"
#include "ControlFlow/C_RandomSelector.h"
#include "ControlFlow/C_Selector.h"
#include "ControlFlow/C_Sequencer.h"

// Student Control Flow Nodes
#include "ControlFlow/C_RandomSequencer.h"

// Example Decorator Nodes
#include "Decorator/D_Delay.h"
#include "Decorator/D_InvertedRepeater.h"
#include "Decorator/D_RepeatFourTimes.h"

// Student Decorator Nodes
#include "Decorator/D_AlwayFail.h"
#include "Decorator/D_AlwaySucceed.h"
#include "Decorator/D_RunTillSuccess.h"
#include "Decorator/D_CheckIfDecided.h"
#include "Decorator/D_RunOnlyOnce.h"
#include "Decorator/D_Inverter.h"
#include "Decorator/D_ChanceOfScrewUp.h"
#include "Decorator/D_CounterToGetFired.h"
#include "Decorator/D_ChanceToMessUpDish.h"
#include "Decorator/D_Timer.h"
#include "Decorator/D_CheckForCustomer.h"

// Example Leaf Nodes
#include "Leaf/L_CheckMouseClick.h"
#include "Leaf/L_Idle.h"
#include "Leaf/L_MoveToFurthestAgent.h"
#include "Leaf/L_MoveToMouseClick.h"
#include "Leaf/L_MoveToRandomPosition.h"
#include "Leaf/L_PlaySound.h"

// Student Leaf Nodes
#include "Leaf/L_ChangeAgentColour.h"
#include "Leaf/L_WalkAcrossStore.h"
#include "Leaf/L_RandomPointOutsideStore.h"
#include "Leaf/L_WalkIntoStore.h"
#include "Leaf/L_GetInLine.h"
#include "Leaf/L_MoveTowardsCashier.h"
#include "Leaf/L_ItsMyTurn.h"
#include "Leaf/L_OrderFood.h"
#include "Leaf/L_ResetCustomerVariables.h"
#include "Leaf/L_EmptyTable.h"
#include "Leaf/L_WalkToTable.h"
#include "Leaf/L_EatAtTable.h"
#include "Leaf/L_HaveItToGo.h"
#include "Leaf/L_LeaveStore.h"
#include "Leaf/L_WaitForCustomer.h"
#include "Leaf/L_TakeOrderFromCustomer.h"
#include "Leaf/L_GetsFired.h"
#include "Leaf/L_AngryCustomer.h"
#include "Leaf/L_OrderMessedUp.h"
#include "Leaf/L_WaitForChef.h"
#include "Leaf/L_RespawnCashier.h"
#include "Leaf/L_GoToKitchen.h"
#include "Leaf/L_CollectFood.h"
#include "Leaf/L_ServeCustomer.h"
#include "Leaf/L_WaitForOrder.h"
#include "Leaf/L_CookDish.h"
#include "Leaf/L_AngryCook.h"
#include "Leaf/L_ServeDish.h"
#include "Leaf/L_CheckOnCashier.h"
#include "Leaf/L_CheckOnChef.h"
#include "Leaf/L_SpawnCustomers.h"