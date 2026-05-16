# IDEA
Adaptated from Google Lens. Since we had the bbox of paragraph level, table level, preproc level,... 

# INSTRUCTION
1) **Understanding**: Read the layout investigation from `explanation.md` to understand the structure of `layout.json`
2) **Text extraction**: Take the content from Level 1 keys,... with attention on their special notation (table, title,...). Note: There is no need to extract equation except it is "inline_equation".
3) **Construct an hiearchy Markdown system**: Based on the instruction from `explanation.md`, design header level (#, ## ...) and concatenation between texts to get better context for my future purposes (translation, under development).
4) **Grouping**: Major text blocks (e.g, title, text in table, para_blocks, discarded_blocks,...) with 100% similarity in content could be grouped for future purposes (translation, under development)
5) **Save links**: Preserve Internal links and Annotation from original PDF to render in Step 6.
6) **Template for rendering**: For each major key (e.g, title, text in table, para_blocks, discarded_blocks,...), convert from layout to Markdown template to render inside provided bbox.
7) **Rendering**: In each page of original PDF, for each major text block (e.g, title, text in table, para_blocks, discarded_blocks,...):
- White cleaning the bbox area of the block
- Choose adaptated size of text to fit perfectly that bbox area 
- Rendering overlapped text in that bbox_area
- **Purpose**: Reserve non-text elements in PDF while rendering translated text elements.

# RULES
Read `CLAUDE.md`
