
9493

Why are SUNITH and NITHIN not labeled as NAME_STUDENT? Mistake or other reason? The numbers next to the names are marked as ID_NUM. It seems like names must be title-cased.

```
USER QUOTES MEANING CONCLUSION

Person 1 SUNITH 522233062166 Aerospace

WHEN THE WELL IS  DRY,WE WILL KNOW THE  WORTH OF WATER.

When a commodity,  article, or other material  object is plentiful, it is  taken for granted. Until  resource is exhausted

Wasting water when we  have plenty and trying to  save it when its low.

Person 2 NITHIN 843756944804 Aerospace

WATER IS LIFE, TAKE CARE  OF IT

Living organisms depend  absolutely on water, which  means that any right to  one's life or the lives of  other, including other  living things, presupposes  a right to water.

By saving water we can  save life.

MORAL OF THE STORY: Water is one of life's most precious resources.  We all must keep it clean.
```


---


7779

Mentions Muhammed Yunus which is a real famous person, and is not considered PII.

Mentions Leroy, then Sullivan Leroy, then Leroy again. Leroy is sometimes B, sometimes I depending on what comes before. 

```
I told them that last year, i was in a village for vacation. I noticed that each  evening, a man named Leroy collected money from  the sellers, traders and farmers of the village.  The amount of the sum collected per person per day  was 1.5 $. Each person could pay his membership  fees daily weekly or monthly. When i discussed with Sullivan Leroy,  he explained me that each six month  he reimburse to each member 90% of his total membership fees and the other 10% is used to face  risks  that could occur in the village. For me, Leroy had create a micro insurance product even he didn’t  call it exactly that. So if Sullivan Leroy can create that kind of product for farmers or sellers, we can do the  same thing with rural population. I also told them the story of Muhammed Yunus who was awarded the  Nobel Peace Prize for founding the Grameen Bank and pioneering the concepts of microcredit and  microfinance. We looked some videos of what Yunus did.  Because a story with no surprise is sometimes  boring, at the end of my story  I presented them Sullivan Leroy whom I had invited.. All of my colleagues  really appreciated Sullivan Leroy participation to the meeting. He shared his experience with us and was  opened to assist us in this project.  At the end of the meeting, we had the feelings that nothing could stop  us.
```

---

4090

Same name twice in a row. First Aakash is B, rest are I


```
Aakash Kumar Aakash Kumar
```


---


7745


The name of the university gets flagged as a NAME_STUDENT. The university is named after a person.

```
Luis Gonzales  Savitribai Phule Pune University Student
```


---


8344

The writer creates made-up people "Tina" and "Marcel" which the model thinks are actual PII. Maybe longer context helps here?


---

21486

The model mistakes "Schlumberger" for a student name. An LLM would know this is a company name.

---

9674

"Alicia Diane Durand" is an author. An LLM should be able to figure this out given the context.

```
We can unlock the power to

think differently by doodling. Alicia Diane Durand’s Discovery Doodles - The complete

series is a good starting point if you are in search of confidence and want to learn basic visual

elements.
```

---

21345

"Ravi" is the name of a business so it isn't PII?

```
During the award ceremony  later in the week I meet with Ravi, professional athlete & seasoned ultra-marathon runner who trains people  to prepare for marathons and Ravi also trains at education centre where school hires his certified trainers  (“Ravi & Co”) service to provide physical fitness training to student in school a part of curriculum.
```

---

19469

This is a story about "Bayo" but maybe this is made-up so it isn't considered PII?

```
In my short life, there are many experiences that could qualify as life-changing.  Every new experience was, at one time or another, the first experience. For good  or bad, each instance changed the course that my life has taken. But, the most  transformative experience was the blind boy next door.

Application

Bayo was a blind boy who hated himself just because he was blind. He hated  everyone, except blessing his loving and pretty girlfriend because Blessing was  always there for him. Bayo said that if he could only see the world he would  marry blessing.
```

---

15045

"Sara" is a persona, so not PII?

```
To create personas I consider Sara. Sara is 34 years old, a busy working mom with a 2 years old son. She  works full time and has no time for shopping. She does most of her shopping online. She enjoys cooking  and likes the healthy eating style. Since she is busy working mom, a suggested shopping list would be  handy for her, the shopping list would be based on her shopping history.
```

---

671

Model thinks `Absträct` is a name. Perhaps due to accent and capitalization. Idea for augmentation?

```
Design Thinking: A Creative Approach to Problem-  Solving

Name: Amparo

[START] Absträct [END] 

Design thinking understanding the human needs related to a problem, reframing the problem in  human-centric ways, creating many ideas in brainstorming sessions, and adopting a hands-on  approach to prototyping and testing-offers a complementary approach to the rational 
```

---

18062

"Maria" is a made up person

```
We choose to create a profile of each person with a lot of information and data about their  activities and the ways of their life.     Aplication>

Name: [START] Maria [END]  Age:37  Labor: Cooker and wife  Sons> 1 girl of 10 years old  She has a little store in the villa area and she sells  empanadas tucumanas.  She want to know how she can growth this business  and do it 
```


## Ideas

- Use LLM to confirm predictions?
  - Is the name "Muhammed Yunus" the name of a student and not a famous person? --> Use Yes/No token probs.
  - Might not need to include the whole essay -- just a 100-200 words before and after.
  - LLM might know about famous people
- Use LLM to make predictions? speed issue?
- Use generic name model to find candidates before passing to another model to confirm/deny?
  - If going for recall, adding all names might be a good first step.
  - Swap out name for generic name?
- Each epoch, swap out the names for completely new ones. Be consistent across the essay.
  - Use a diverse range of names from across cultures.
- whenever a B-NAME_STUDENT is found, automatically check next token
- whenever an I-NAME_STUDENT is found, check the previous token
- Find all words that are title cased and longer than 2 characters, see if they are in name bank (be sure to normalize accents)
  - In the provided set, the following code produces 163888 matches. There are a total of 4992533 tokens in the training set. (3% matches)

```python
  matches = list()

for d in data:
    matches.extend(list(set([x for x in d["tokens"] if x.istitle() and len(x) > 2])))

len(matches) # 163888
``` 