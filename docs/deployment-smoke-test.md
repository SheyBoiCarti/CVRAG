# Streamlit Deployment Smoke Test

Run this checklist against the deployed Streamlit Community Cloud application after every release.

- [ ] Cold start completes without exposing an exception or API key.
- [ ] The page has no sidebar, framework developer controls, or re-index action.
- [ ] Summary, Skills, Experience, and Education suggestions produce streamed answers.
- [ ] A free-form question produces citations with source and page.
- [ ] “What did he do there?” works after a role-specific question.
- [ ] An unsupported question says the information is not listed in the CV.
- [ ] Citation text containing HTML-like characters renders literally.
- [ ] Clear conversation removes prior messages.
- [ ] The layout works at 375px width and with reduced motion enabled.
- [ ] A missing or invalid secret produces a friendly error without internal details.
- [ ] Refreshing and opening a second session do not rebuild the index unnecessarily.

Do not add the deployed URL to the README until every applicable item passes.
