Only two street addresses were provided. They both have:

1. American addresses
2. "\n" in middle
3. Apartment numbers


One is in the header, one is in the intro paragraph.

---

## Rules

Given this, there are some possibilities for rules:

1. Search for two-letter state acronym (capitalized) and 5 numbers afterwards. Maybe a comma in between
2. Search for "Apt. \d+\n"
3. Make sure it starts with

---

## Ideas

- Should probably vary the locale in faker to make the model robust to a variety of formats. US, Canada, UK, France, Spain, Germany, India, China, Japan, Brazil, Saudi Arabia, Israel, Turkey
- Include PO Box 
- Include newlines

---


9854

There is a newline character in the middle that frequently does not get predicted.

```
591 Smith Centers Apt. 656
Joshuamouth, RI 95963
```


---

11442

There is also a newline in the middle of the address.

```
743 Erika Bypass Apt. 419
Andreahaven, IL 54207
```