"""Prompt templates for different prompting approaches."""
from typing import Protocol

from finetune import coa


class BaseTemplate(Protocol):
    def get_prompt(self, instance, coa_to_bodyid: dict) -> str:
        raise NotImplementedError(type(self))


class OneStepSubCategoryClassification(BaseTemplate):
    """Classify account from limited set of options."""
    target_column = "account_code"
    prompt_template = """
You are an experienced accountant whose responsibility is to associate expense related to acquired product/service to a correct account.

You have the following information available:
 
===
Name of the company that acquired the product/service:
{receiver_name}

Business line of the company that acquired the product/service:
{main_line_business}

Business line of the invoice sender:
{invoice_sender_main_business_line}

Name of the acquired product/service:  
{product_name}

Number of items acquired:
{amount} {unit}

Total price:
{total_price}

Invoice comes from:  
{invoice_sender}
===

The expense should be posted to one of the following accounts:

===
{categories_str}
===

YOUR TASKS:

- Think hard what is the correct account the expense should be posted to.
- Use all pieces of expense information.
- The purpose for which the product/service was acquired is the key to which account the expense should be posted.
- Consider most logical option based on the expense and general accounting principles.
- Answer only with the account code for the correct account.

account code:
"""

    def get_prompt(self, instance, coa_to_bodyid: dict):
        full_coa = coa_to_bodyid[str(instance["receiver_id"])]
        categories_str, number_of_targets = coa.get_categories_and_leaf_count(
            instance, full_coa
        )
        prompt = self.prompt_template.format(
            receiver_name=instance["receiver_name"],
            main_line_business=instance["reveiver_main_line_business"],
            product_name=instance["product_name"],
            amount=f"{instance['amount']:.2f}",
            total_price=f"{instance['amount'] * instance['price']:,.2f} {instance['currency']}",
            unit=instance["unit"],
            categories_str=categories_str,
            invoice_sender=instance["invoice_sender"],
            invoice_sender_main_business_line=instance[
                "invoice_sender_main_business_line"
            ],
        )
        instance["prompt"] = prompt
        instance["categories"] = categories_str
        instance["num_targets"] = number_of_targets
        return instance

    def get_target(self, instance, all_coa_elements) -> str:
        target = str(instance["account_code"])
        return target


# class OneStepSubCategoryClassificationCreateReasoningStep(BaseTemplate):
#     """Prompt for reasoning for chain-of-thought approach."""
#     target_column = "reasoning"
#     prompt_template = """
# You are an experienced accountant whose responsibility is to associate expense related to acquired product/service to a correct account.

# You have the following information available:

# ===

# Name of the company that acquired the product/service:
# {receiver_name}

# Business line of the company that acquired the product/service:
# {main_line_business}

# Business line of the invoice sender:
# {invoice_sender_main_business_line}

# Name of the acquired product/service:  
# {product_name}

# Number of items acquired:
# {amount} {unit}

# Total price:
# {total_price}

# Invoice comes from:  
# {invoice_sender}

# ===

# Question: To which account in the Chart Of Accounts excerpt the expense should be posted?

# {categories_str}

# ===

# YOUR TASKS:

# - You know the final answer is {account_code}
# - Provide chain of thought reasoning that will help to provide the final answer at the very last line of your answer.
# - Use all pieces of expense information.
# - The purpose for which the product/service was acquired is the key to which account the expense should be posted.
# - Let figuring out the purpose of the expense guide your reasoning.
# - Only provide the final answer in the last line of your output
# - Do not repeat the list of accounts in the Chart of Accounts
# - Always include the sentence "Therefore the account code is: {account_code}" as the final line of your answer.
# """

#     def get_prompt(self, instance, coa_to_bodyid: dict):
#         """_summary_

#         Args:
#             instance (_type_): _description_
#             full_coa (_type_): dict of CoAs for the different bodyids.

#         Returns:
#             _type_: _description_
#         """
#         full_coa = coa_to_bodyid[str(instance["receiver_id"])]
#         categories_str, number_of_targets = coa.get_categories_and_leaf_count(
#             instance, full_coa
#         )
#         prompt = self.prompt_template.format(
#             receiver_name=instance["receiver_name"],
#             main_line_business=instance["reveiver_main_line_business"],
#             product_name=instance["product_name"],
#             amount=f"{instance['amount']:.2f}",
#             total_price=f"{instance['amount'] * instance['price']:,.2f} {instance['currency']}",
#             unit=instance["unit"],
#             categories_str=categories_str,
#             invoice_sender=instance["invoice_sender"],
#             invoice_sender_main_business_line=instance[
#                 "invoice_sender_main_business_line"
#             ],
#             account_code=instance["account_code"],
#         )
#         instance["prompt"] = prompt
#         instance["categories"] = categories_str
#         instance["num_targets"] = number_of_targets
#         return instance

#     def get_target(self, instance, all_coa_elements) -> str:
#         target = str(instance["account_code"])
#         return target


# class OneStepSubCategoryClassificationUtilizeReasoningStep(BaseTemplate):
#     """Utilizing the trained reasoning steps."""
#     prompt_template = """
# You are an experienced accountant whose responsibility is to associate expense related to acquired product/service to a correct account.

# You have the following information available:
 
# ===
# Name of the company that acquired the product/service:
# {receiver_name}

# Business line of the company that acquired the product/service:
# {main_line_business}

# Business line of the invoice sender:
# {invoice_sender_main_business_line}

# Name of the acquired product/service:  
# {product_name}

# Number of items acquired:
# {amount} {unit}

# Total price:
# {total_price}

# Invoice comes from:  
# {invoice_sender}
# ===

# The expense should be posted to one of the following accounts:

# ===
# {categories_str}
# ===

# YOUR TASKS:

# - Provide chain of thought reasoning that will help to provide the final answer at the very last line of your answer.
# - Use all pieces of expense information.
# - The purpose for which the product/service was acquired is the key to which account the expense should be posted.
# - Let figuring out the purpose of the expense guide your reasoning.
# - Only provide the final answer in the last line of your output
# - Do not repeat the list of accounts in the Chart of Accounts
# - Always include the sentence "Therefore the account code is: [account code]" as the final line of your answer.

# """

#     def get_prompt(self, instance, coa_to_bodyid: dict):
#         full_coa = coa_to_bodyid[str(instance["receiver_id"])]
#         categories_str, number_of_targets = coa.get_categories_and_leaf_count(
#             instance, full_coa
#         )
#         prompt = self.prompt_template.format(
#             receiver_name=instance["receiver_name"],
#             main_line_business=instance["reveiver_main_line_business"],
#             product_name=instance["product_name"],
#             amount=f"{instance['amount']:.2f}",
#             total_price=f"{instance['amount'] * instance['price']:,.2f} {instance['currency']}",
#             unit=instance["unit"],
#             categories_str=categories_str,
#             invoice_sender=instance["invoice_sender"],
#             invoice_sender_main_business_line=instance[
#                 "invoice_sender_main_business_line"
#             ],
#         )
#         instance["prompt"] = prompt
#         instance["categories"] = categories_str
#         instance["num_targets"] = number_of_targets
#         return instance
